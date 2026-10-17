"""Persistent owner of the frozen graph-half candidate; inference stays on owner thread."""
import json
from pathlib import Path
from baseline import load_baseline
from candidate import Candidate
from lut_ops import Tables
from mlp_graph import HalfGraph, lib
from mlp_subgraph import install

class Runtime:
    def __init__(self):
        self.candidate = None
        self.restore_graph = None
        self.close_model = None
        self.closed = False
        try:
            self.model, self.engine, self.base_counts, self.close_model = load_baseline(library='/work/libpipeline_bridge.so')
            cfg = json.loads(Path('/service/config.json').read_text())
            self.candidate = Candidate(self.model, self.engine, Tables())
            self.candidate.use(cfg)
            class OwnedGraph(HalfGraph):
                def __init__(self, path):
                    super().__init__(path)
                    if lib.mlp_graph_set_core(self.handle, cfg.get('graph_core') or 0):
                        self.close()
                        raise RuntimeError('Graph core mask rejected')
            self.restore_graph, self.graph_counts = install(self.model, OwnedGraph)
        except BaseException:
            self.close()
            raise

    def counts(self):
        return {'cpu': {k: dict(v) for k, v in self.candidate.counts.items()},
                'graphs': dict(self.graph_counts)}

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.restore_graph:
                self.restore_graph()
        finally:
            try:
                if self.candidate:
                    self.candidate.reset()
            finally:
                if self.close_model:
                    self.close_model()
