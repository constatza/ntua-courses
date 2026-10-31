\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ntuabeamer}[2026/03/24 NTUA Beamer class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax

\LoadClass{beamer}
\RequirePackage{ntuabeamer}
