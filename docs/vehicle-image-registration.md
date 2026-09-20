# Vehicle image registration

## Recommended automated workflow

Manually maintain only the images below `img/input/<Size>/`. Generate all
`data/projects/vehicles_<Size>.csv` files from the master dimension table with:

```powershell
python scripts/sync_project_csvs.py
python scripts/sync_project_csvs.py --apply
```

The first command is a dry run. The second writes the project CSV files. New
image filenames should equal `DIMENSION-ID` with the final
market suffix (such as `US`) omitted. Word order, punctuation, and case do not
matter. Existing project rows are retained as confirmed image aliases, which
supports abbreviated filenames such as `JL WRANGLER 2DR.webp`.

If an image year spans multiple dimension rows with different measurements, or
the model/year is absent from the master table, the row is still written with
`name`, `Size`, and `image_path`; its three dimension fields remain blank. The
processing pipeline skips incomplete rows until authoritative dimensions are
added to the master table and synchronization is run again.

After adding, replacing, moving, or deleting an input image, run the sync again.
The image's parent directory supplies `Size`; `L-MM`, `W-MM`, and `H-MM` always
come from `data/dimensions/全尺码全量.csv`. Image files are never changed.

Generation can perform the same synchronization first:

```powershell
python -m pickup_measure.main --sync-vehicles --continue
```

The older one-at-a-time registration workflow remains available below.

Use `scripts/add_vehicle_image.py` for an existing image. Its filename (without
the extension) is the canonical vehicle `name`; the image is never renamed or
moved. The script derives `image_path`, checks the Size group and dimensions,
prevents a duplicate `name`/`Size` pair, and defaults to a dry run.

Example:

```powershell
python scripts/add_vehicle_image.py `
  --size "3XL+" `
  --length-mm 5184 --width-mm 1928 --height-mm 1565 `
  --source "img/input/3XL+/2009 Lincoln MKS-V6 Sedan.png" `
  --apply
```

The canonical rule is `name = <image filename without extension>` and
`image_path = img/input/<Size>/<image filename>`. Run without `--apply` first
when reviewing a proposed match.

Before generating output, synchronize every CSV record to this rule, without
touching image files, by running:

```powershell
python scripts/sync_vehicle_names.py --apply
```

The processing flow can execute this step automatically before generation:

```powershell
python -m pickup_measure.main --sync-vehicles --continue
```

Rows with missing image files are reported and retained unchanged.

For a missing path, the sync step searches only its Size directory and applies
the path/name update only when there is exactly one semantic match. A candidate
with a year inside the record's year range is valid; no candidate or multiple
candidates leaves the CSV row unchanged and prevents output generation.

The `Maybach` trim and generic `Sedan` body-style tokens are treated as
compatible with an otherwise matching Mercedes-Benz S-Class image.

For semantic matching, an image whose year is inside a record's year range may
represent that record when the remaining make/model text matches. For example,
`2007 BMW 7 Series.png` is a valid representative image for a `2002-2008 BMW 7
Series` record. Once matched, use the actual image filename as both `name` and
the final component of `image_path`.

## Generated-image cache

`python -m pickup_measure.main --continue` uses SHA-256 of the image referenced
by `image_path`, not the generated filename, to decide whether a record can be
skipped. Successful runs persist the hash in
`data/image_generation_hashes.json`. Deterministic Qwen response-validation
failures are cached there as well, including the error and pipeline stage. If
the image hash and record ID are unchanged, `--continue` skips the failed item
without calling Qwen again; cached failures remain failures in the run summary
because no output was generated. Transient API, connection, and timeout errors
are not cached. To force analysis of one record, run it without `--continue`,
for example:

```powershell
python -m pickup_measure.main --only-name "2007 BMW 7 Series"
```
