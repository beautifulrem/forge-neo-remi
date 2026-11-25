import { describe, expect, it } from "vitest"
import { isImageFile, looksLikeImage, MAX_INFLATED_BYTES, readExifUserComment, readImageParameters, readPngText } from "./read-image-info"

async function deflate(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Response(bytes as BodyInit).body!.pipeThrough(new CompressionStream("deflate"))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

const latin1 = (text: string) => Uint8Array.from(text, (c) => c.charCodeAt(0))
const utf8 = (text: string) => new TextEncoder().encode(text)
const concat = (...parts: Uint8Array[]) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0))
  let at = 0
  for (const part of parts) {
    out.set(part, at)
    at += part.length
  }
  return out
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const head = new Uint8Array(8)
  new DataView(head.buffer).setUint32(0, data.length)
  head.set(latin1(type), 4)
  return concat(head, data, new Uint8Array(4)) // CRC is not checked
}

const SIGNATURE = Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)
const png = (...chunks: Uint8Array[]) => concat(SIGNATURE, chunk("IHDR", new Uint8Array(13)), ...chunks, chunk("IEND", new Uint8Array()))

const PARAMS = "1girl, 笑顔\nNegative prompt: lowres\nSteps: 20, Seed: 42"

describe("readPngText", () => {
  it("reads tEXt, iTXt (plain and compressed) and zTXt chunks", async () => {
    const bytes = png(
      chunk("tEXt", concat(latin1("Software\0"), latin1("Remi"))),
      chunk("iTXt", concat(latin1("parameters\0"), Uint8Array.of(0, 0), latin1("\0\0"), utf8(PARAMS))),
      chunk("iTXt", concat(latin1("comment\0"), Uint8Array.of(1, 0), latin1("en\0Comment\0"), await deflate(utf8("zipped ✓")))),
      chunk("zTXt", concat(latin1("legacy\0"), Uint8Array.of(0), await deflate(latin1("old text")))),
    )
    expect(await readPngText(bytes)).toEqual({ Software: "Remi", parameters: PARAMS, comment: "zipped ✓", legacy: "old text" })
  })

  it("inflates only the wanted keys, and refuses oversized text", async () => {
    const bomb = await deflate(new Uint8Array(MAX_INFLATED_BYTES + 1))
    const bytes = png(
      chunk("zTXt", concat(latin1("other\0"), Uint8Array.of(0), bomb)),
      chunk("zTXt", concat(latin1("parameters\0"), Uint8Array.of(0), bomb)),
      chunk("tEXt", concat(latin1("parameters\0"), latin1("fallback"))),
    )
    // The oversized chunk is skipped; the next one with the key is read.
    expect(await readPngText(bytes, ["parameters"])).toEqual({ parameters: "fallback" })
  })

  it("returns nothing for a non-PNG", async () => {
    expect(await readPngText(latin1("GIF89a"))).toEqual({})
  })
})

/** A TIFF block (little-endian "II" unless said) with IFD0 → Exif IFD → UserComment. */
function tiffWithComment(comment: Uint8Array, little = true): Uint8Array {
  const header = 8
  const ifd0 = header // 2 + 12 + 4 = 18 bytes
  const exifIfd = ifd0 + 18 // 18 bytes
  const value = exifIfd + 18
  const out = new Uint8Array(value + comment.length)
  const v = new DataView(out.buffer)
  out.set(latin1(little ? "II*\0" : "MM\0*"), 0)
  v.setUint32(4, ifd0, little)
  v.setUint16(ifd0, 1, little)
  v.setUint16(ifd0 + 2, 0x8769, little)
  v.setUint16(ifd0 + 4, 4, little)
  v.setUint32(ifd0 + 6, 1, little)
  v.setUint32(ifd0 + 10, exifIfd, little)
  v.setUint16(exifIfd, 1, little)
  v.setUint16(exifIfd + 2, 0x9286, little)
  v.setUint16(exifIfd + 4, 7, little)
  v.setUint32(exifIfd + 6, comment.length, little)
  v.setUint32(exifIfd + 10, value, little)
  out.set(comment, value)
  return out
}

function utf16be(text: string): Uint8Array {
  const out = new Uint8Array(text.length * 2)
  for (let i = 0; i < text.length; i++) {
    out[i * 2] = text.charCodeAt(i) >> 8
    out[i * 2 + 1] = text.charCodeAt(i) & 0xff
  }
  return out
}

function utf16le(text: string): Uint8Array {
  const out = new Uint8Array(text.length * 2)
  for (let i = 0; i < text.length; i++) {
    out[i * 2] = text.charCodeAt(i) & 0xff
    out[i * 2 + 1] = text.charCodeAt(i) >> 8
  }
  return out
}

const hex = (text: string) => Uint8Array.from(text.match(/../g)!, (byte) => parseInt(byte, 16))
/** The bytes piexif.dump writes (modules/images.py, JPEG and WebP) for the comment "猫耳少女，微笑". */
const PIEXIF_CJK = hex(
  "4578696600004d4d002a00000008000187690004000000010000001a000000000001928600070000001600000028554e49434f444500732b80335c115973ff0c5fae7b11",
)

describe("EXIF UserComment", () => {
  it("decodes piexif's UNICODE (UTF-16BE) comments", () => {
    expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), utf16be(PARAMS))))).toBe(PARAMS)
  })

  // The byte order is not in the format: piexif writes big-endian, other tools little-endian.
  it("reads either byte order, whatever the first character", () => {
    const cjk = "猫耳少女，微笑\nSteps: 20, Sampler: Euler a"
    const quoted = "“quoted” start\nSteps: 20"
    for (const text of [PARAMS, cjk, quoted]) {
      for (const little of [true, false]) {
        expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), utf16be(text)), little))).toBe(text)
        expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), utf16le(text)), little))).toBe(text)
      }
    }
    // No Latin text to tell by: the TIFF block's own order
    expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), utf16be("猫耳少女")), false))).toBe("猫耳少女")
    expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), utf16le("猫耳少女")), true))).toBe("猫耳少女")
    // A byte-order mark decides
    expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), Uint8Array.of(0xff, 0xfe), utf16le("猫耳")), false))).toBe("猫耳")
    expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), Uint8Array.of(0xfe, 0xff), utf16be("猫耳")), true))).toBe("猫耳")
  })

  // A writer that pads the comment to an even count, or cuts it short, leaves a lone byte: it is
  // not half a character to show as U+FFFD
  it("drops a lone last byte", () => {
    for (const little of [true, false]) {
      expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), utf16be(PARAMS), Uint8Array.of(0)), little))).toBe(PARAMS)
      expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), utf16le(PARAMS), Uint8Array.of(0x20)), little))).toBe(PARAMS)
      expect(readExifUserComment(tiffWithComment(concat(latin1("UNICODE\0"), Uint8Array.of(0xfe, 0xff), utf16be("猫耳"), Uint8Array.of(0)), little))).toBe("猫耳")
    }
  })

  it("reads a comment piexif wrote", async () => {
    const app1 = concat(Uint8Array.of(0xff, 0xe1, (PIEXIF_CJK.length + 2) >> 8, (PIEXIF_CJK.length + 2) & 0xff), PIEXIF_CJK)
    const jpeg = concat(Uint8Array.of(0xff, 0xd8), app1, Uint8Array.of(0xff, 0xda, 0, 2, 0xff, 0xd9))
    expect(await readImageParameters(new Blob([jpeg]))).toBe("猫耳少女，微笑")
  })

  it("decodes ASCII comments", () => {
    expect(readExifUserComment(tiffWithComment(concat(latin1("ASCII\0\0\0"), latin1("Steps: 5"))))).toBe("Steps: 5")
  })

  it("finds the comment in a JPEG APP1 segment", async () => {
    const exif = concat(latin1("Exif\0\0"), tiffWithComment(concat(latin1("UNICODE\0"), utf16be(PARAMS))))
    const app1 = concat(Uint8Array.of(0xff, 0xe1, (exif.length + 2) >> 8, (exif.length + 2) & 0xff), exif)
    const jpeg = concat(Uint8Array.of(0xff, 0xd8), app1, Uint8Array.of(0xff, 0xda, 0, 2, 0xff, 0xd9))
    expect(await readImageParameters(new Blob([jpeg]))).toBe(PARAMS)
  })

  it("finds the comment in a WebP EXIF chunk", async () => {
    const exif = tiffWithComment(concat(latin1("UNICODE\0"), utf16be("Steps: 8")))
    const size = new Uint8Array(4)
    new DataView(size.buffer).setUint32(0, exif.length, true)
    const body = concat(latin1("WEBP"), latin1("EXIF"), size, exif, exif.length % 2 ? new Uint8Array(1) : new Uint8Array())
    const riffSize = new Uint8Array(4)
    new DataView(riffSize.buffer).setUint32(0, body.length, true)
    expect(await readImageParameters(new Blob([concat(latin1("RIFF"), riffSize, body)]))).toBe("Steps: 8")
  })

  it("prefers the PNG parameters chunk", async () => {
    expect(await readImageParameters(new Blob([png(chunk("tEXt", concat(latin1("parameters\0"), latin1("Steps: 3"))))]))).toBe("Steps: 3")
    expect(await readImageParameters(new Blob([png()]))).toBeNull()
  })
})

describe("image signatures", () => {
  it("knows the formats the server reads and refuses other files", async () => {
    const riff = (kind: string) => concat(latin1("RIFF"), new Uint8Array(4), latin1(kind))
    const iso = (brand: string) => concat(new Uint8Array(4), latin1("ftyp"), latin1(brand))
    for (const head of [SIGNATURE, Uint8Array.of(0xff, 0xd8, 0xff, 0xe0), latin1("GIF89a"), latin1("BM...."), riff("WEBP"), latin1("II*\0"), latin1("MM\0*"), iso("avif"), iso("heic")]) {
      expect(looksLikeImage(head), Array.from(head).join(",")).toBe(true)
    }
    for (const head of [latin1("hello world"), latin1("%PDF-1.7"), riff("WAVE"), iso("isom"), new Uint8Array()]) {
      expect(looksLikeImage(head), Array.from(head).join(",")).toBe(false)
    }
    expect(await isImageFile(new Blob([png() as BlobPart]))).toBe(true)
    expect(await isImageFile(new Blob(["not an image"], { type: "image/png" }))).toBe(false)
  })
})

