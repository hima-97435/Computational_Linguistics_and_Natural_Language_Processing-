# Computational Linguistics & Natural Language Processing (NLP) Lab

Welcome to the **Computational Linguistics and Natural Language Processing (NLP) Laboratory** repository. This repository is dedicated to exploring the core foundations of data analysis, exploratory data visualization, and fundamental text preprocessing pipelines in NLP. 

The codebase is split into three primary laboratories:
1. **Lab 1**: Exploratory Data Analysis (EDA) and Visualization on Tabular Datasets.
2. **LAB 2**: NLP Text Preprocessing, Word/Sentence Tokenization (using NLTK, spaCy, and Regex), and Stop Words Filtering.
3. **Lab 3**: Stemming, Lemmatization, and Regular Expression Pattern Matching.

---

## 📂 Project Structure

Below is the directory structure of the project. Note that environment settings, debug files, and editor configurations are excluded via Git tracking.

```text
NLP Lab/
├── .gitignore                   # Configured to exclude venv, .vscode, main.py, main.dSYM
├── README.md                    # Main documentation file (this file)
├── Lab1/
│   ├── Lab1.ipynb               # Jupyter Notebook containing Experiments 1.1 to 1.10
│   └── employee_information_100.csv # Raw employee database for analysis (100 rows)
├── LAB2/
│   ├── Lab2.ipynb               # Jupyter Notebook containing Experiments 2.1 to 2.3
│   ├── 2.1_text_data.txt        # Input dataset for preprocessing
│   ├── 2.2_tokenization_data.txt# Input text dataset for word and sentence tokenization
│   └── 2.3_clean_data.txt       # Input text dataset for stop words removal
└── Lab3/
    └── lab3.ipynb               # Jupyter Notebook containing Experiments 3.1 to 3.3
```

---

## 🔬 Lab Details

### 📊 Lab 1: Exploratory Data Analysis & Visualization
This lab focuses on data cleaning, grouping, statistical analysis, and visual storytelling using **Pandas** and **Matplotlib**. It serves as a foundational step for understanding data distributions before feeding them into computational pipelines.

#### Experiments Covered:
*   **Experiment 1.1: Average Salary by Department**
    *   *Concept*: Data grouping and aggregation.
    *   *Implementation*: Compute average salaries per department and display them using a matplotlib bar chart.
*   **Experiment 1.2: Department Headcount**
    *   *Concept*: Categorical distributions.
    *   *Implementation*: Display employee headcount across departments using value counts.
*   **Experiment 1.3: Gender Diversity Analysis**
    *   *Concept*: Proportion representation.
    *   *Implementation*: Render a pie chart displaying the percentage breakdown of male vs. female employees.
*   **Experiment 1.4: Salary Distribution Histogram**
    *   *Concept*: Continuous variables distributions.
    *   *Implementation*: Draw a histogram to identify skewness, range, and common salary bands.
*   **Experiment 1.5: Experience vs. Salary Scatter Plot**
    *   *Concept*: Correlation and regression insights.
    *   *Implementation*: Map professional experience (years) against salary to visualize compensation growth patterns.
*   **Experiment 1.6: Top Earners Identification**
    *   *Concept*: Sorting and filtering.
    *   *Implementation*: Sort dataset by salary descending to extract the top 10 highest-paid employees.
*   **Experiment 1.7: Departmental Peak Compensation**
    *   *Concept*: Grouped max-aggregation.
    *   *Implementation*: Identify the maximum salary ceiling within each department.
*   **Experiment 1.8: Above-Average Earners**
    *   *Concept*: Threshold filtering.
    *   *Implementation*: Compute global average salary and filter list of employees earning above it.
*   **Experiment 1.9: Average Tenure by Department**
    *   *Concept*: Departmental experience profiles.
    *   *Implementation*: Compute the mean years of experience grouped by department.
*   **Experiment 1.10: Age Distribution Histogram**
    *   *Concept*: Demographic distribution.
    *   *Implementation*: Plot age frequency distribution using a histogram.

---

### 🔤 LAB 2: Computational Linguistics & NLP Preprocessing
This lab introduces the fundamental processing layers required for any Natural Language Processing pipeline. Text data is inherently unstructured; preprocessing cleanses and breaks text down into tokens suitable for language modeling.

#### 1. Experiment 2.1: Basic Text Preprocessing Pipeline
Text cleansing is essential to reduce vocabulary size and normalize characters. This experiment builds a pipeline that performs:
1.  **Lowercasing**: Normalizes casing (e.g., "Apple" and "apple" become identical) while counting uppercase letters.
2.  **Punctuation Removal**: Strips structural symbols (e.g., `!`, `,`, `.`) and outputs the count of unique punctuation characters removed.
3.  **Number Removal**: Strips numerical characters (`0-9`) to isolate linguistic components.
4.  **Whitespace Normalization**: Collapses multiple spaces, tabs, or newlines down to a single space.
5.  **Output Visualisation**: Displays side-by-side comparisons of raw and cleaned texts.

#### 2. Experiment 2.2: Tokenization (Word & Sentence Level)
Tokenization is the task of breaking down a block of text into individual linguistic units (tokens) such as words, phrases, or sentences. This experiment contrasts three different technical implementations:
*   **NLTK (Natural Language Toolkit)**:
    *   Uses rule-based algorithms like `PunktSentenceTokenizer` for sentence splitting and Penn Treebank-style rules for word tokenization.
*   **spaCy**:
    *   An industrial-strength library that utilizes a state-of-the-art dependency parsing pipeline. It models tokenization using language-specific rules and non-destructive tokenizer graphs.
*   **Custom Regular Expressions (Python `re`)**:
    *   A zero-dependency approach using regex patterns (e.g., splitting on punctuation followed by spaces `(?<=[.!?])\s+`) to demonstrate the inner workings of tokenizers.

#### 3. Experiment 2.3: Tokenization & Stop Words Removal
Stop words (e.g., "is", "the", "a", "an", "to") carry high frequency but low semantic information. Removing them helps algorithms focus on content words (nouns, verbs, adjectives).
*   **Implementation**:
    *   Reads `2.3_clean_data.txt`.
    *   Tokenizes the input string.
    *   Filters tokens against the NLTK stop words dictionary.
    *   Outputs the original tokens, filtered tokens, and isolates the stop words that were stripped out.

---

### 🧬 Lab 3: Stemming, Lemmatization & Regular Expressions
This lab explores text normalization through stemming and lemmatization, along with pattern extraction using regular expressions. These techniques are essential for reducing vocabulary complexity and extracting structured information from unstructured text.

#### 1. Experiment 3.1: Stemming using Porter Stemmer
Stemming reduces words to their root form by stripping suffixes using heuristic rules. The Porter Stemmer is one of the most widely used stemming algorithms in information retrieval.
*   **Input Words**: playing, played, plays, studies, studying, connected, connection, computers
*   **Implementation**:
    *   Uses NLTK's `PorterStemmer` to apply suffix-stripping rules.
    *   Displays original words alongside their stemmed forms.
*   **Key Insight**: Stemming is fast but may produce non-dictionary roots (e.g., "studies" → "studi").

#### 2. Experiment 3.2: Lemmatization using WordNet Lemmatizer
Lemmatization reduces words to their dictionary base form (lemma) using vocabulary analysis and morphological rules, requiring part-of-speech (POS) tagging for accuracy.
*   **Input Words**: cats, dogs, running, runs, ran, studies, studying, better, children, mice, went, ate, leaves, caring
*   **Implementation**:
    *   Uses NLTK's `WordNetLemmatizer` with manually assigned POS tags (noun, verb, adjective).
    *   Maps each word to its correct lemma (e.g., "mice" → "mouse", "better" → "good").
*   **Key Insight**: Unlike stemming, lemmatization produces valid dictionary words and handles irregular forms (e.g., "went" → "go").

#### 3. Experiment 3.3: Regex-Based Information Extraction
Regular expressions provide a powerful pattern-matching mechanism for extracting structured data from unstructured text. This experiment extracts emails, URLs, mobile numbers, hashtags, and mentions.
*   **Input Text**: A sample workshop announcement containing various contact and social media references.
*   **Extraction Patterns**:
    *   **Email Addresses**: Pattern matching `@` domain structures.
    *   **URLs**: Detecting `http://`, `https://`, and `www.` prefixed links.
    *   **Mobile Numbers**: Indian mobile number formats with optional country code (+91).
    *   **Hashtags**: Words prefixed with `#` (e.g., #NLP, #Python).
    *   **Mentions**: Usernames prefixed with `@` (e.g., @NLPWorkshop).

---
