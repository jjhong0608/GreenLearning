%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Coupled class for LaTeX files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}

\DeclareOption{10pt}{\OptionNotUsed}
\DeclareOption{11pt}{\OptionNotUsed}
\DeclareOption{twocolumn}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax
\LoadClass[11pt]{article}

% -------------------------------------------------
% Packages
% -------------------------------------------------
\RequirePackage{calc}
\RequirePackage{indentfirst}
\RequirePackage{fancyhdr}
\RequirePackage{newtxtext,newtxmath}
\RequirePackage[
  a4paper,
  top=1cm,
  bottom=2.5cm,
  left=2.75cm,
  right=2.45cm,
  headheight=40pt,
  headsep=18pt,
  includehead
]{geometry}
\linespread{1}

% -------------------------------------------------
% No hyphenation
% -------------------------------------------------
\hyphenpenalty=10000
\exhyphenpenalty=10000
\tolerance=1000
\emergencystretch=1em


% -------------------------------------------------
% keywords
% -------------------------------------------------
\newcommand{\keywords}[1]{%
  \gdef\@keywords{#1}
}

% -------------------------------------------------
% Header
% -------------------------------------------------
\pagestyle{fancy}
\fancyhf{}

\fancyhead[R]{%
\fontsize{10}{12}\selectfont
\bfseries
\begin{tabular}{r}
Digital Twins in Engineering\\
DTE 2027\\
March 8--12, 2027, Yokohama, Japan
\end{tabular}
}

\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% -------------------------------------------------
% Title data
% -------------------------------------------------
\def\author{\@ifnextchar[{\@getshortauthor}{\@getauthor}}

\def\@getauthor#1{%
  \gdef\@author{#1}%
  \gdef\@shortauthor{#1}%
}

\def\@getshortauthor[#1]#2{%
  \gdef\@shortauthor{#1}%
  \gdef\@author{#2}%
}


% -------------------------------------------------
% Title layout
% -------------------------------------------------
\newcommand{\address}[1]{\gdef\@address{#1}}
\renewcommand{\maketitle}{%
  \newpage
  \thispagestyle{fancy}%
  \vspace*{0pt}%
  \begin{center}%
    \vspace*{-12pt}%
    {\fontsize{14}{14}\selectfont\bfseries \@title\par}%
    \vspace{12pt}%
    {\fontsize{12}{12}\selectfont\bfseries \@author\par}%
    \vspace{7pt}%
    {\fontsize{11}{11}\selectfont
      \def\and{\par\vspace{7pt}}%
      \@address\par}%
  \end{center}%
  \vspace{12pt}%
  \noindent  
  {\fontsize{11}{12}\selectfont\bfseries Key Words:\ }
  {\fontsize{11}{12}\selectfont\itshape \@keywords\par}
  \vspace{12pt}%
}

% -------------------------------------------------
% Abstract
% -------------------------------------------------
\renewenvironment{abstract}{%
  \par
  \vspace{4pt}%
  \begin{centering}
    {\fontsize{12}{12}\selectfont\bfseries\ ABSTRACT\par}
  \end{centering}
  \vspace{6pt}%
  \begingroup
  \fontsize{12}{12}\selectfont
  \setlength{\parindent}{0pt}%
  \setlength{\parskip}{6pt plus 1pt minus 1pt}%
}{%
  \par
  \endgroup
  \vspace{12pt}%
}

% -------------------------------------------------
% Bibliography
% -------------------------------------------------
\renewenvironment{thebibliography}[1]{%
  \vspace{12pt}%
  \begin{centering}%
  {\fontsize{12}{12}\selectfont\bfseries\ REFERENCES\par}%
  \vspace{4pt}%
  \end{centering}%
  \list{\@biblabel{\@arabic\c@enumiv}}{%
    \settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \usecounter{enumiv}%
    \renewcommand\theenumiv{\@arabic\c@enumiv}%
    \setlength{\itemsep}{0pt}%
    \setlength{\parsep}{0pt}%
  }%
  \sloppy
  \clubpenalty4000
  \widowpenalty4000%
  \sfcode`\.\@m
}{%
  \endlist
}

% -------------------------------------------------
% General spacing
% -------------------------------------------------
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\AtBeginDocument{%
  \maketitle
  \let\maketitle\relax
}

\endinput