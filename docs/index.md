# SPARTA documentation

- [good_practices.md](good_practices.md) — a checklist for an analysis, in
  the order the work is done: what to do, what goes wrong if you don't, and
  where it is explained.
- [conventions.md](conventions.md) — units, the ln λ grid, velocity
  conventions, air/vacuum, times, native sampling, preprocessing order, and
  when the error bars are calibrated.
- [correlation_engines.md](correlation_engines.md) — UNICOR and TODCOR: the
  CCF and the 2-D fit, their correlation and log-likelihood outputs, RVs,
  errors and result keys, side by side.
- [tiravel.md](tiravel.md) — TIRAVEL: what is implemented, how to read ρ
  and the weights, and the plan for the velocity optimiser.
- [periodograms.md](periodograms.md) — GLS, PDC, error-aware PDC, USuRPER and
  the shift/shape partial periodograms; result keys and p-values.
- [instruments.md](instruments.md) — every survey key: frames, time systems,
  native sampling, recommended settings (generated).
- [open_api.md](open_api.md) — the open surface: every atom of `sparta`,
  `unicor` and `todcor` an analysis can be built from, with its import path,
  units and assumptions.
- [api.md](api.md) — signatures and docstrings of the public API of `sparta`,
  `unicor` and `todcor` (generated).
- [limitations.md](limitations.md) — known limitations, with their size and
  where they are discussed.
- [roadmap.md](roadmap.md) — a proposed roadmap, for the maintainers to edit.

The TODCOR derivation is in
[packages/todcor/docs/todcor_method.md](../packages/todcor/docs/todcor_method.md).

These pages are the reference: each convention, formula and limitation is
stated here once. The notebooks in [examples/](../examples/README.md) link to
them: the how-tos (`examples/howto/`) are short recipes with the objects, and
the tutorials (`examples/tutorials/`) build each method from the atoms.
