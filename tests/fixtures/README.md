# Test fixtures

- `sample_data.csv`, `tricky_data.csv`: tiny hand-written tables covering one
  column of each semantic type (identifier, datetime, temporal, geographic,
  ordinal, boolean, text, ...).
- `penguins.csv`: Palmer Penguins (Gorman, Williams & Fraser, 2014), CC0, from
  https://github.com/allisonhorst/palmerpenguins, as written by
  `python -m evaluation.real_world.download_datasets`. Vendored so the
  real-world regression test runs without a download.
