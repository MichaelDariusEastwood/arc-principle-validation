# How to verify the priority claims yourself

Everything below is checkable without contacting the author. If any step fails, the claim fails.

## 1. Verify the timestamped manuscripts (the core anchors)

Each priority `.eml` in this component is the exact file whose SHA-256 was published. Recompute it:

```bash
shasum -a 256 "01-dec8-2024-priority-anchor.eml"
# expect: f0d1f38ffd8546152d9d9d28dc5ec083c16a35858f2c12b63e69db7ed50901ad

shasum -a 256 "02-apr30-2025-book-manuscript.eml"
# expect: 09f5b5e156ed96f8883eaf668495fd350898ce62be6294b8f788e0e2d6dcb664
```

If the numbers match, the file has not changed since it was hashed and published.

## 2. Confirm the emails are genuine Gmail-origin, self-addressed timestamps

Open each `.eml` in any text editor and read the headers:
- `Date:` — the timestamp (8 Dec 2024 02:45:18 +0000 / 30 Apr 2025 13:37:05 +0100).
- `Message-ID:` — begins `CAGPsKA…`, the Gmail-origin pattern.
- `From:`/`To:` — self-addressed (author to himself). The manuscript content is in the attachments.
- `DKIM-Signature:` — Google's cryptographic signature over the message as sent; it binds the
  content and the date to Google's mail servers at send time.

## 3. Confirm the deposit dates (not the creation dates)

On OSF, open any canonical file and click **Revisions**. Version 1 of the earliest-deposited
files is dated 17 March 2026 — the deposit date. The *creation* date is the manuscript anchor in step 1.

## 4. Confirm the book

ISBN 978-1806056200 — check any public bookseller or national deposit library for the publication date.

---
The point of this folder is that none of the above requires trusting the author: the hashes are
reproducible, the DKIM signatures are Google's, and the deposit dates are OSF's. Verify, don't believe.
