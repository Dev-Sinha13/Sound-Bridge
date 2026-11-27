# Sound Bridge interface

The interface is a local audio workspace. Search has the primary visual emphasis; collection setup and encoder experiments have their own views. The existing forest-green identity remains in the navigation rail and controls, with a light workspace for readable file names, waveforms and tables.

The user's [UI tools collection](https://github.com/maxbogo/awesome-ai-tools-for-ui) informed the review. Relevant references were [Impeccable](https://impeccable.style/) for hierarchy and reducing visual clutter, and [Vercel's Web Interface Guidelines](https://github.com/vercel-labs/web-interface-guidelines) for semantic controls, focus, forms, empty/loading states, and responsive layout. No external UI service or runtime dependency was added.

- Search: a collection picker, large query field, example prompts, and collapsed filters. Results retain actual similarity values and timestamps; waveform amplitude is normalized for display, not a confidence visualization. Only one audio player runs at a time. Relevance judgment buttons are disclosed on demand.
- Collections: actual model identity, root folder, file/window counts, and compatibility-aware search actions. Indexing options sit behind an advanced disclosure. Compatible collections appear first.
- Alignment lab: native and bridge comparisons remain distinct. Score summaries use the validation-selected artifact, or the native result, never the best test row. Complete tables remain available on demand, with explicit synthetic labeling.
- Models: a native modal dialog keeps configuration out of the main search surface. The model remains visible in the desktop rail; a mobile toolbar button exposes the same settings.
- Interaction: collection options are updated only when the collection list changes, avoiding interruption of an open picker. Jobs have loading/cancellation states. The `/` shortcut focuses search; the tab list supports arrow/Home/End keys, visible focus and selected state. Dialog Escape/focus behavior uses native HTML.
- Responsive layout: a fixed rail on desktop becomes top navigation below 700 px. Forms and library/lab columns stack before they become cramped. No external fonts or assets are needed, and reduced-motion preferences disable animated spinners.

This redesign changes presentation and task organization. It does not change model weights, retrieval metrics, index formats, or the interpretation of similarity scores. Collection names and query text are inserted as text, not HTML. Existing APIs, CSRF protection, local-only inference, and model compatibility checks remain in place.
