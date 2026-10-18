# Codebook demo: design notes

The page is **an open code book lying on a desk**: a bound book with two pages per spread and a thumb index of coloured tabs on its fore-edge. It is not a scrolling page. The tabs turn between four spreads:

| Tab | Left page | Right page |
|---|---|---|
| Send | title, introduction, message slip, sample messages | the message transcribed into code words by each book |
| Entries | the first forty learned entries, set like a dictionary (number, phrase, leader dots, count) | the longest entries, and a note on training speed |
| Rates | bytes per token against book size (chart) | the tiktoken check, and encoding speed |
| Notes | three short notes | the rate table and the colophon |

The concept comes from telegraph code books, which replaced common phrases with single code words to cut the per-word charge. BPE does the same.

## Tokens (defined at the top of `template.html`)

- **Desk:** green-black baize around the book (`#2f3b33`).
- **Pages:** cream (`#f4eedd`), with a darker gutter shading down the spine and a few stacked page edges under the book.
- **Ink:** warm black for text, with one red (`#a8321d`) for entry numbers, emphasis and the drop cap.
- **Tabs:** red, blue, ochre and slate, all checked for 4.5:1 contrast with their cream labels.
- **Lamp off (dark mode):** dark pages on a near-black desk, with the same layout. The tabs are darkened to keep their label contrast.
- **Type:**
  - Libre Caslon Display for headings, and Libre Caslon Text for reading.
  - Small caps for running heads.
  - Courier Prime for anything that is a code word: tokens, ids, counts, the message slip.
- **Chart colours:** IMDB is blue (`#2a5d92` / `#4f8fd6` dark) and Rust is red (`#b2361f` / `#e0694f`). Both pass the dataviz palette check in light and dark. Each series also has its own line style and marker (solid with circles, dashed with squares), and OpenAI's books are hollow rings.

## Pieces

- **Running heads and folios** on every page, as in a printed book.
- **Tokens as tape strips:** alternating tints, each with its id printed small above it. Spaces show as a raised dot, line breaks as ↵, and a token that is only part of a character shows its bytes in hex.
- **Below 980px** the book becomes a single column, and the thumb tabs become a sticky row of tabs along the top edge.

## Things to avoid

- Turning it back into a long scrolling page.
- More than one accent colour on the page itself (the tabs are the exception).
- Card shadows inside the book.
- Any real company's branding.
