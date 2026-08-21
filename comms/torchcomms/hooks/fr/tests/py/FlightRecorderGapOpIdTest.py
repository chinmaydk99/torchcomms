#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Regression test for the FlightRecorder ring-buffer slot desync.

``FlightRecorder::record()`` keyed its ring-buffer write slot off the
*global* ``op_id`` (``op_id % max_entries_``), while every read path
(``getEntry``, ``retire_id``, ``update_state``) derives the slot from the
recorder-local, per-epoch ``id_`` via ``getIdxFromId``. The two agree only
when the recorder observes every op-id consumption starting at 0.

Any op that consumes a global op-id without being recorded leaves a gap:
every collective consumes an id unconditionally (``TorchComm.cpp``), but a
communicator with no hook attached is never recorded. A recorder whose
first entry then lands on a nonzero op-id desyncs writes from reads and
trips the growth-phase assertion ``entries_.size() == idx + 1`` (the same
failure mode as SplitGroupTest under ``TORCHCOMM_FR_BUFFER_SIZE > 0``).

This test exercises the exact sequence: run one collective on an *unhooked*
comm (silently consumes op-id 0), then record collectives from a hooked
comm. Pre-fix, the first hooked collective trips the growth-phase
assertion. Post-fix, the recorder slots by its local ``id_`` and both
collectives land contiguously.
"""

import json
import os
import unittest
from datetime import timedelta

import torch
import torchcomms
from torchcomms.hooks import FlightRecorderHook


class TestFlightRecorderGapOpId(unittest.TestCase):
    backend = os.environ["TEST_BACKEND"]
    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    def test_gap_from_unhooked_comm_does_not_desync(self) -> None:
        # Isolated instance resets the global op-id generator so the first
        # collective below deterministically consumes op-id 0.
        recorder = FlightRecorderHook(max_entries=100, isolated=True)

        # --- Unhooked comm: consumes op-id 0 without recording it. ---
        comm_a = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="fr_gap_op_id_a",
            timeout=timedelta(seconds=300),
        )
        t = torch.rand(4, device=self.device)
        comm_a.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
        self.assertEqual(recorder.size(), 0)

        # --- Hooked comm: first recorded op has global op-id 1. Pre-fix,
        # record() writes at slot 1 % max_entries_ while entries_ is empty,
        # tripping TORCH_CHECK(entries_.size() == idx + 1). ---
        comm_b = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="fr_gap_op_id_b",
            timeout=timedelta(seconds=300),
        )
        recorder.register_with_comm(comm_b)
        comm_b.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
        self.assertEqual(recorder.size(), 1)

        # Steady state after a nonzero start: slots stay contiguous.
        comm_b.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
        self.assertEqual(recorder.size(), 2)

        dump = json.loads(recorder.dump_json())
        entries = dump["entries"]
        self.assertEqual(len(entries), 2)
        op_ids = sorted(entry["op_id"] for entry in entries)
        self.assertEqual(
            op_ids,
            [1, 2],
            f"expected contiguous op-ids [1, 2], got {op_ids}",
        )

        comm_a.finalize()
        comm_b.finalize()


if __name__ == "__main__":
    unittest.main()
