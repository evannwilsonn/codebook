# Codebook demo: design notes

The page is **a page from an old telegraph code book**, matching the project thumbnail: aged paper with faint foxing, a double-ruled border, a letterspaced CODEBOOK masthead with an italic subtitle over a double hairline rule, two columns split by a hairline, and a footer line of stats in small caps. It is not a scrolling page. A thumb index cut into the fore-edge turns between four spreads:

| Tab | Left page | Right page |
|---|---|---|
| Send | title, introduction, message slip, sample messages | the message transcribed into code words by each book |
| Entries | the first forty learned entries, set like a dictionary (number, phrase, leader dots, count) | the longest entries, and a note on training speed |
| Rates | bytes per token against book size (chart) | the tiktoken check, and encoding speed |
| Notes | three short notes | the rate table and the colophon |

The concept comes from telegraph code books, which replaced common phrases with single code words to cut the per-word charge. BPE does the same.

## Tokens (defined at the top of `template.html`)

- **Paper (light):** `#eee2c4` with a few foxing spots (`#b08a4a` at low opacity) and a soft darker edge.
- **Ink:** `#2b2117` for text and rules; rules in the table are `#b9a57c` dotted leaders.
- **Red:** `#a3302a`, the one accent: entry ids, emphasis, the drop cap, the selected tab, and the one highlighted row (a 12% red tint), as in the thumbnail.
- **Lamp off (dark mode):** a dark ink ground (`#1d1813`) with paper-coloured type (`#eee2c4`). The red stays `#a3302a` for fills and the selected tab; red text lightens to `#e2796b` to keep 4.5:1.
- **Type (self-hosted in `docs/fonts/`, SIL OFL):**
  - Young Serif for the masthead and headings, letterspaced wide on the masthead.
  - Crimson Pro for reading, italic for subtitles, emphasis and folios ("p. 8").
  - IBM Plex Mono for anything that is a code word (tokens, ids, counts, the message slip) and for small-caps labels.
- **Chart colours:** IMDB is ink and Rust is red. Each series also has its own line style and marker (solid with circles, dashed with squares), and OpenAI's books are hollow rings.

## Pieces

- **Running heads and folios** on every page, as in a printed book, plus a stats footer under each spread.
- **Tokens as tape strips:** alternating tints, each with its id printed small above it. Spaces show as a raised dot, line breaks as ↵, and a token that is only part of a character shows its bytes in hex.
- **Below 980px** the book becomes a single column, and the thumb tabs become a sticky row of tabs along the top edge.

## Things to avoid

- Turning it back into a long scrolling page.
- More than one accent colour.
- Card shadows inside the book.
- Any real company's branding.
