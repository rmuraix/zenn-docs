# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a [Zenn](https://zenn.dev/) article repository for publishing Japanese technical articles. Content lives in `articles/` (individual articles) and `books/` (book-format content). Articles are written in Japanese Markdown with a YAML frontmatter header.

## Commands

```bash
npm ci                  # Install dependencies
npm run preview         # Start local preview server (zenn preview)
npm run new             # Create a new article (zenn new:article)
npm run lint:check      # Check Markdown with textlint (CI uses this)
npm run lint:fix        # Auto-fix textlint issues
npm run format:check    # Check formatting with Prettier (CI uses this)
npm run format:write    # Auto-fix formatting with Prettier
```

CI runs `format:check` then `lint:check` on every push/PR to `main`.

## Article Format

Each article file under `articles/` requires this frontmatter:

```yaml
---
title: "記事タイトル"
emoji: "🔥"
type: "tech"       # tech: 技術記事 / idea: アイデア
topics: ["tag1", "tag2"]
published: true    # false to keep as draft
---
```

Article filenames are auto-generated hashes (e.g., `22c256b41754be.md`) — do not rename them.

## Linting Rules

textlint enforces Japanese writing conventions via two presets:

- **`preset-ja-technical-writing`** — Japanese technical writing style (periods allowed: `:`)
- **`preset-jtf-style`** — JTF (Japan Translation Federation) style guide

To suppress a lint rule for a specific range, use `<!-- textlint-disable --> ... <!-- textlint-enable -->` comments. The allowlist also permits `:::details ...` headings (Zenn's collapsible block syntax).

Prettier formats all `.md`, `.json`, and `.yaml` files. The `.prettierignore` excludes `node_modules`, `.git`, and lock/package files.

## Zenn-specific Markdown Extensions

Zenn supports custom syntax beyond standard Markdown:

- `:::message` / `:::message alert` — info/warning callout boxes
- `:::details <title>` — collapsible sections
- Bare URLs on their own line render as embed cards
