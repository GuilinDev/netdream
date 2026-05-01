// Palette aligned with paper figures (paper/figures/fig_overview.pdf)
// Pale pastel watercolor + accent blue.

export const palette = {
  // Pastel fills
  paleBlue: "#DCE9F4",
  paleLavender: "#ECE3F0",
  paleMint: "#E8F2E7",
  palePink: "#FDEEF3",
  palePeach: "#FFF1D6",
  paleYellow: "#FFF8E0",
  paleCoral: "#FBE3E2",

  // Panel backgrounds (even paler)
  panelBlueBg: "#F2F6FB",
  panelLavBg: "#F7F2F9",
  panelMintBg: "#F2F8F1",

  // Accent (use sparingly)
  accentBlue: "#5B8DBE",
  accentBlueDeep: "#3F6FA0",

  // Status / risk indicators
  riskLow: "#A3D9A5",   // green
  riskMid: "#F5E1A4",   // yellow
  riskHigh: "#E89999",  // coral red

  // Text + structure
  textPrimary: "#222222",
  textSecondary: "#444444",
  textMuted: "#888888",
  borderGray: "#999999",
  borderLight: "#CCCCCC",

  // Background gradient
  bgTop: "#FAFCFF",
  bgBottom: "#EFF4F9",
} as const;

export type PaletteColor = keyof typeof palette;
