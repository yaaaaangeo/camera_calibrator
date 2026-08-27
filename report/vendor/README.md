# Vendored assets

`plotly-gl3d.min.js` is the official minified "gl3d" partial distribution
bundle of [plotly.js](https://github.com/plotly/plotly.js) (MIT license,
see `plotly-gl3d.min.js.LICENSE`), pulled from the `plotly.js-gl3d-dist-min`
npm package. The gl3d partial bundle supports exactly the trace types the
interactive 3D viewer needs (`scatter3d`, `mesh3d`) at roughly a third the
size of the full plotly.js bundle (~1.7 MB vs. ~4.9 MB minified).

It's vendored (checked into the repo) rather than loaded from a CDN at
report-view time so `report.html` stays fully self-contained and works
offline -- this matters for CI pipelines that may not have general
internet egress, and for anyone archiving/sharing a report without also
needing network access to view the interactive scene later.

To update the version: `npm pack plotly.js-gl3d-dist-min`, extract the
tarball, and replace `plotly-gl3d.min.js` (+ its LICENSE) with the new
one.
