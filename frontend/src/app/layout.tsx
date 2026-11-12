import type { Metadata } from "next";
import { Instrument_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// Instrument Sans for text, JetBrains Mono for scores and ranks.
const instrument = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-instrument",
  display: "swap",
});
const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Seafloor Anomaly Review",
  description: "Survey review tool for seafloor debris and anomaly detection",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${instrument.variable} ${jetbrains.variable}`}>
      <body>{children}</body>
    </html>
  );
}