Core System v4, generated 2026-10-29 by tools/export.js. Do not edit; edit project/tokens.json and re-run.

Copy this folder into a project (for example static/core/) and load, in this order:
  <link rel="stylesheet" href="core/tokens.css">
  <link rel="stylesheet" href="core/components/bundle.css">
  <script src="core/components/bundle.js"></script>
(or core/core.css instead of the two stylesheets). Maps: add core/leaflet.css after Leaflet's own CSS.
Single-file exports (one HTML file you send or open from disk) can use core/tokens-embedded.css instead of tokens.css:
it carries the fonts inside (about 1.2 MB), so nothing else has to travel with the page; tokens-embedded-lab.css
(about 0.6 MB) leaves out the serif, for Lab and console pages.
Migrating from ui-tokens.css: load bridge-ui-tokens.css after tokens.css, instead of ui-tokens.css.
MapLibre: maplibre.css after maplibre-gl.css. Leaflet and MapLibre themselves are not in the kit: vendor them.
Everything works offline: fonts are in fonts/ with their OFL licences; icons are Lucide (LICENSE-lucide.txt).
palette.json holds every colour resolved per theme, for canvas, WebGL and Python.
See the Core System guide 'Adopting Core System' for the rest.
