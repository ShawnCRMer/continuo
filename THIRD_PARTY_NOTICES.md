# Third-party notices

Continuo is MIT licensed (`LICENSE`). It contains or is derived from the following third-party work. Found by
scanning `backend/film-suite`, `desktop/film-suite/src`, `workflows` and the bundled `desktop/plugin.js`
(esbuild input list) for vendored packages, licence headers and porting notes.

## Bundled into `desktop/plugin.js`

| Package | Version | Licence | Copyright |
|---|---|---|---|
| @codemirror/state | 6.7.6 | MIT | Copyright (C) 2018-2021 by Marijn Haverbeke <marijn@haverbeke.berlin> and others |
| @codemirror/view | 6.43.14 | MIT | Copyright (C) 2018-2021 by Marijn Haverbeke <marijn@haverbeke.berlin> and others |
| @codemirror/commands | 6.11.1 | MIT | Copyright (C) 2018-2021 by Marijn Haverbeke <marijn@haverbeke.berlin> and others |
| @codemirror/language | 6.13.1 | MIT | Copyright (C) 2018-2021 by Marijn Haverbeke <marijn@haverbeke.berlin> and others |
| @lezer/common | 1.5.3 | MIT | Copyright (C) 2018 by Marijn Haverbeke <marijn@haverbeke.berlin> and others |
| @lezer/highlight | 1.2.5 | MIT | Copyright (C) 2018 by Marijn Haverbeke <marijn@haverbeke.berlin> and others |
| style-mod | 4.1.4 | MIT | Copyright (C) 2018 by Marijn Haverbeke <marijn@haverbeke.berlin> and others |
| w3c-keyname | 2.2.8 | MIT | Copyright (C) 2016 by Marijn Haverbeke <marijn@haverbeke.berlin> and others |
| crelt | 1.0.7 | MIT | Copyright (C) 2020 by Marijn Haverbeke <marijn@haverbeke.berlin> |
| @marijn/find-cluster-break | 1.0.4 | MIT | Copyright (C) 2024 by Marijn Haverbeke <marijn@haverbeke.berlin> |
| three | 0.180.0 | MIT (three.js) | Copyright © 2010-2025 three.js authors |

React and `@hermes/plugin-sdk` are provided by Hermes at run time and are not bundled.

## Icons and artwork

- **Lucide** icons (undo-2, redo-2, focus, ellipsis, map-pin, scan-text) in `src/shared/ui/icons.js`: ISC License.
  Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT); all other
  copyright (c) Lucide Contributors 2022.
- **Hermes mark** (`nous-girl-dark.png`, embedded in `src/shared/ui/hermes-mark-data.js`): MIT License,
  Copyright (c) 2025 Nous Research.

## Data

- **Krea moodboard style prompts** (`dashboard/image_builder/style-boards.v1.json.gz`, 3549 style prompts: title,
  prompt, keywords, negative): taken from `data/krea_moodboards_slim.json` of
  [Andro-Meta/ComfyUI-Krea-Moodboards](https://github.com/Andro-Meta/ComfyUI-Krea-Moodboards) at commit
  `4b439f9c6eb092dd0b8e36e466e7ec9bf2cb7ba7`: MIT License, Copyright (c) 2026 ComfyUI Krea Moodboards contributors.
  That project describes the data as Krea's public moodboard catalog and states it is unofficial and not affiliated
  with Krea AI; neither is Continuo. Preview images are not shipped; the style browser loads them from Krea's image
  host (`optim-images.krea.ai`) when you open a board.

## Ported code

- **FilmCraft** (storytold/filmcraft), edit crate (`crates/edit/src/lib.rs`: insert, overwrite, split, delete, trim,
  roll, slip) and interchange crate (FCPXML mapping): MIT OR Apache-2.0. Ported to Python and JavaScript in
  `dashboard/timeline/ops.py`, `dashboard/timeline_export/` and `src/timeline/ops/`.

## Workflow templates

- **ComfyUI workflow templates** (`dashboard/workflows/flux2-klein-9b-kv-edit-v1`, `flux3-klein-sd-edit-v1`,
  `minimax-h3-t2v`): MIT License, Copyright (c) 2023-present Comfy Org. The licence text ships next to each workflow
  as `COMFYUI-WORKFLOW-TEMPLATES-LICENSE.txt`.

## Design references, not shipped

OpenCut (ripple behaviour), Final Draft, DaVinci Resolve and other tools are named in code comments as behaviour
references only; no code from them is included.

## Python packages installed by Hermes

numpy (BSD-3-Clause), librosa (ISC), soundfile (BSD-3-Clause), httpx (BSD-3-Clause), Pillow (MIT-CMU) are declared in `plugin.yaml` and
installed by Hermes' package manager into its dependency environment; they are not part of this package.

## Models

No model weights ship with Continuo. Each model you install has its own licence; `dashboard/workflows/model-sources.json`
records the licence named on each model's card.

## MIT License text (for the MIT-licensed components above)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## ISC License text (Lucide)

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted,
provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
OF THIS SOFTWARE.
