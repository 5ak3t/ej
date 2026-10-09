"""ejpack v1 CONTAINER: one binary file = fixed prefix + JSON header + 64-byte aligned raw sections.

Layout (all integers little-endian):
  bytes 0-7    MAGIC b'EJPACK1\\n'
  bytes 8-15   uint64 header length H
  bytes 16-47  sha256 of the header bytes
  bytes 48..   the header: UTF-8 JSON {'format': 'ejpack-v1', 'align': 64, 'sections': {name: {'offset', 'length', 'sha256'}},
               'meta': {...}}; offsets are relative to DATA = align_up(48 + H)
  DATA..       the sections, each starting at a multiple of ALIGN (zero padding between them).
Reading verifies the header digest and the sha256 of every section before anything is decoded; no pickle anywhere."""
import hashlib
import json
import struct

MAGIC = b'EJPACK1\n'
FORMAT = 'ejpack-v1'
ALIGN = 64
PREFIX = 48


class PackError(ValueError):
    """A file that is not a valid ejpack v1 container (bad magic, digest or section)."""


def align_up(n, a=ALIGN):
    """n rounded up to a multiple of a."""
    return (n + a - 1) // a * a


def sha256(b):
    """Hex sha256 of a bytes-like object."""
    return hashlib.sha256(b).hexdigest()


def write(path, sections, meta):
    """Write sections (ordered list of (name, bytes)) and meta (JSON-able dict) to path; returns the header dict."""
    names = [n for n, _ in sections]
    if len(set(names)) != len(names):
        raise PackError(f'duplicate section names: {sorted(n for n in names if names.count(n) > 1)}')
    table, off = {}, 0
    for name, data in sections:
        table[name] = {'offset': off, 'length': len(data), 'sha256': sha256(data)}
        off = align_up(off + len(data))
    header = {'format': FORMAT, 'align': ALIGN, 'sections': table, 'meta': meta}
    hb = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    data0 = align_up(PREFIX + len(hb))
    with open(path, 'wb') as f:
        f.write(MAGIC + struct.pack('<Q', len(hb)) + hashlib.sha256(hb).digest() + hb)
        f.write(b'\0' * (data0 - PREFIX - len(hb)))
        pos = 0
        for name, data in sections:
            f.write(b'\0' * (table[name]['offset'] - pos))
            f.write(data)
            pos = table[name]['offset'] + len(data)
    return header


def parse(buf):
    """(header dict, data start) of a container held in buf; checks magic, length and header digest."""
    if len(buf) < PREFIX or bytes(buf[:8]) != MAGIC:
        raise PackError('not an ejpack v1 file (bad magic)')
    (hlen,) = struct.unpack('<Q', bytes(buf[8:16]))
    hb = bytes(buf[PREFIX:PREFIX + hlen])
    if len(hb) != hlen or hashlib.sha256(hb).digest() != bytes(buf[16:48]):
        raise PackError('header digest mismatch')
    header = json.loads(hb)
    if header.get('format') != FORMAT:
        raise PackError(f'format {header.get("format")!r}, expected {FORMAT!r}')
    return header, align_up(PREFIX + hlen, header.get('align', ALIGN))


def content_digest(header):
    """Hex sha256 over what decoding reads: every section's sha256 and length, and the header meta except 'source' (where the
    pack was written from). Two packs of the same model written from different source files share this digest."""
    meta = {k: v for k, v in header['meta'].items() if k != 'source'}
    secs = {n: [s['sha256'], s['length']] for n, s in header['sections'].items()}
    return sha256(json.dumps({'meta': meta, 'sections': secs}, sort_keys=True, separators=(',', ':')).encode())


class Reader:
    """A loaded container: .header, .meta, .file_sha256, .digest (content_digest) and section(name) -> memoryview (every
    section verified at open)."""

    def __init__(self, path):
        with open(path, 'rb') as f:
            self.buf = memoryview(f.read())
        self.file_sha256 = sha256(self.buf)
        self.header, self.data0 = parse(self.buf)
        self.digest = content_digest(self.header)
        self.meta = self.header['meta']
        for name, s in self.header['sections'].items():
            v = self._view(s)
            if len(v) != s['length'] or sha256(v) != s['sha256']:
                raise PackError(f'section {name!r}: sha256 / length mismatch')

    def _view(self, s):
        a = self.data0 + s['offset']
        return self.buf[a:a + s['length']]

    def section(self, name):
        """The bytes of section name (a read-only memoryview into the file buffer)."""
        if name not in self.header['sections']:
            raise PackError(f'missing section {name!r}')
        return self._view(self.header['sections'][name])

    def sizes(self):
        """{section name: length in bytes}."""
        return {n: s['length'] for n, s in self.header['sections'].items()}

    def close(self):
        """Drop the file buffer (sections handed out before must have been copied)."""
        try:
            self.buf.release()
        except BufferError:  # a section view is still referenced somewhere; the buffer goes with it
            pass
