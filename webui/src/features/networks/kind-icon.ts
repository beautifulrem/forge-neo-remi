import { Box, Contrast, type IconComponent, Layers, Type } from "@/components/icons"

/** A missing preview's glyph per network kind; never the Generate button's wand (`Sparkles`). */
export const KIND_ICON: Record<string, IconComponent> = { lora: Layers, embeddings: Type, checkpoints: Box, vae: Contrast, text_encoders: Type }
