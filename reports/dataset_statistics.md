# GeoFinBERT Dataset Statistics Report

## 1. Corpus Overview
- **Historical Range**: 2000-01-01 to 2026-11-11
- **Unlabeled DAPT Corpus (Stage 1)**: 14,486 articles
- **Human-Labeled Sentiment Corpus (Stage 2)**: 6,809 articles
- **Data Integrity Guarantee**: 100% genuine real-world articles. Zero synthetic text, zero model-generated pseudo-labels.
- **Syndication Deduplication**: Removed exact duplicate hashes and near-duplicate wire syndications.

## 2. Topic Distribution
### Stage 1: Unlabeled DAPT Articles
| Research Topic | Article Count | Share (%) |
| :--- | :---: | :---: |
| `gold_bullion_markets` | 8,549 | 59.02% |
| `us_iran_relations` | 4,031 | 27.83% |
| `oil_crude_markets` | 1,588 | 10.96% |
| `middle_east_geopolitics` | 318 | 2.20% |

### Stage 2: Human-Labeled Sentiment Articles
| Research Topic | Article Count | Share (%) |
| :--- | :---: | :---: |
| `gold_bullion_markets` | 6,768 | 99.40% |
| `oil_crude_markets` | 41 | 0.60% |

## 3. Human Sentiment Label Distribution (Stage 2)
| Sentiment Class | Label ID | Count | Share (%) |
| :--- | :---: | :---: | :---: |
| **Positive** | `0` | 3,481 | 51.12% |
| **Negative** | `1` | 3,009 | 44.19% |
| **Neutral** | `2` | 319 | 4.68% |

## 4. Annual Article Coverage (2000–2026)
| Year | DAPT Articles | Human-Labeled Articles | Total |
| :---: | :---: | :---: | :---: |
| 2000 | 23 | 22 | 45 |
| 2001 | 29 | 28 | 57 |
| 2002 | 51 | 41 | 92 |
| 2003 | 127 | 110 | 237 |
| 2004 | 226 | 202 | 428 |
| 2005 | 233 | 196 | 429 |
| 2006 | 338 | 287 | 625 |
| 2007 | 216 | 207 | 423 |
| 2008 | 224 | 216 | 440 |
| 2009 | 239 | 221 | 460 |
| 2010 | 279 | 251 | 530 |
| 2011 | 615 | 531 | 1,146 |
| 2012 | 540 | 412 | 952 |
| 2013 | 610 | 442 | 1,052 |
| 2014 | 505 | 430 | 935 |
| 2015 | 398 | 343 | 741 |
| 2016 | 845 | 728 | 1,573 |
| 2017 | 1,644 | 1,460 | 3,104 |
| 2018 | 1,074 | 672 | 1,746 |
| 2019 | 17 | 10 | 27 |
| 2026 | 6,253 | 0 | 6,253 |

## 5. Accredited News Publishers and Wire Sources
### Top Publishers in Unlabeled DAPT Corpus
| Publisher / Source | Article Count | Share (%) |
| :--- | :---: | :---: |
| `marketwatch` | 4,412 | 30.46% |
| `financial_press` | 2,007 | 13.85% |
| `forexlive` | 1,046 | 7.22% |
| `economictimes` | 924 | 6.38% |
| `moneycontrol` | 899 | 6.21% |
| `zerohedge` | 521 | 3.60% |
| `benzinga` | 506 | 3.49% |
| `cnbc` | 232 | 1.60% |
| `rttnews` | 212 | 1.46% |
| `globenewswire` | 174 | 1.20% |