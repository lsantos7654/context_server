"""LLM-based content analysis service for intelligent metadata extraction."""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from .content_analysis import CodeBlock, ContentAnalysis

logger = logging.getLogger(__name__)


class LLMAnalysisService:
    """Service for LLM-powered content analysis and metadata extraction."""

    def __init__(self, llm_service=None):
        """Initialize with optional LLM service dependency."""
        self.llm_service = llm_service
        self.fallback_enabled = True

    async def analyze_content_intelligently(
        self, content: str, url: Optional[str] = None
    ) -> ContentAnalysis:
        """Perform intelligent content analysis using LLM with fallback to regex patterns."""

        try:
            # Primary: LLM-based analysis
            if self.llm_service:
                llm_analysis = await self._llm_content_analysis(content, url)
                if llm_analysis:
                    logger.info("Successfully completed LLM-based content analysis")
                    return llm_analysis

            # Fallback: Enhanced pattern-based analysis (better than current hardcoded approach)
            logger.info("Using fallback enhanced pattern analysis")
            return await self._enhanced_fallback_analysis(content, url)

        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return self._create_minimal_analysis(content)

    async def _llm_content_analysis(
        self, content: str, url: Optional[str] = None
    ) -> Optional[ContentAnalysis]:
        """Perform LLM-based content analysis with structured prompts."""

        # Skip LLM analysis if service is not available
        if not self.llm_service:
            logger.info("LLM service not initialized, skipping LLM analysis")
            return None
        
        if not hasattr(self.llm_service, 'available') or not self.llm_service.available:
            logger.info("LLM service not available (no API key), skipping LLM analysis")
            return None

        try:
            # Phase 1: Content classification and metadata extraction
            classification_result = await self._classify_content_with_llm(content)

            # Phase 2: Programming language and code analysis
            code_analysis_result = await self._analyze_code_with_llm(content)

            # Phase 3: Concept and relationship extraction
            concept_result = await self._extract_concepts_with_llm(content)

            # Phase 4: Quality and complexity assessment
            quality_result = await self._assess_quality_with_llm(content)

            # Combine all LLM results into ContentAnalysis
            return self._combine_llm_results(
                content,
                url,
                classification_result,
                code_analysis_result,
                concept_result,
                quality_result,
            )

        except Exception as e:
            logger.warning(f"LLM content analysis failed: {e}")
            return None

    async def _classify_content_with_llm(self, content: str) -> Dict:
        """Use LLM to classify content type and extract basic metadata."""

        prompt = f"""Analyze this documentation content and classify it. Return a JSON response with these fields:

content_type: One of ["tutorial", "api_reference", "code_example", "troubleshooting", "concept_explanation", "general"]
primary_purpose: Brief description of what this content is meant to teach or explain
summary: 2-3 sentence summary of the content
content_category: ["documentation", "code", "mixed", "reference"]
target_audience: ["beginner", "intermediate", "advanced", "mixed"]

Content to analyze:
{content[:2000]}...

Return only valid JSON:"""

        try:
            response = await self.llm_service.generate_response(prompt)
            if not response or not response.strip():
                logger.warning("LLM classification returned empty response")
                return {}
            
            # Clean response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM classification failed - invalid JSON: {e}. Response: {response[:200] if response else 'Empty'}")
            return {}
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return {}

    async def _analyze_code_with_llm(self, content: str) -> Dict:
        """Use LLM to analyze programming content and extract code metadata."""

        # Extract code blocks first using regex
        code_blocks = self._extract_markdown_code_blocks(content)

        if not code_blocks:
            return {"code_percentage": 0, "primary_language": None, "code_blocks": []}

        # Analyze code blocks with LLM
        code_analysis_prompt = f"""Analyze these code blocks and return JSON with:

primary_language: The main programming language used (or null if mixed/unclear)
code_percentage: Estimated percentage of content that is code (0-100)
programming_patterns: Array of programming concepts found (e.g., "async", "error_handling", "oop", "functional")
complexity_level: ["simple", "moderate", "complex"]
code_quality_indicators: Array of quality aspects (e.g., "well_commented", "good_naming", "follows_conventions")

Code blocks to analyze:
{self._format_code_blocks_for_llm(code_blocks)}

Return only valid JSON:"""

        try:
            response = await self.llm_service.generate_response(code_analysis_prompt)
            if not response or not response.strip():
                logger.warning("LLM code analysis returned empty response")
                return {
                    "code_percentage": len(code_blocks) * 10,
                    "code_blocks": code_blocks,
                }
            
            # Clean response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            code_analysis = json.loads(cleaned_response)
            code_analysis["code_blocks"] = code_blocks  # Add extracted blocks
            return code_analysis
        except json.JSONDecodeError as e:
            logger.warning(f"LLM code analysis failed - invalid JSON: {e}. Response: {response[:200] if response else 'Empty'}")
            return {
                "code_percentage": len(code_blocks) * 10,
                "code_blocks": code_blocks,
            }
        except Exception as e:
            logger.warning(f"LLM code analysis failed: {e}")
            return {
                "code_percentage": len(code_blocks) * 10,
                "code_blocks": code_blocks,
            }

    async def _extract_concepts_with_llm(self, content: str) -> Dict:
        """Use LLM to extract key concepts, topics, and relationships."""

        prompt = f"""Extract key concepts and topics from this content. Return JSON with:

key_concepts: Array of 5-10 most important technical concepts or topics
topic_keywords: Array of 10-15 searchable keywords that represent this content
api_references: Array of API methods, functions, or endpoints mentioned
technologies_mentioned: Array of technologies, frameworks, or tools referenced
learning_objectives: Array of what someone would learn from this content

Content to analyze:
{content[:2000]}...

Return only valid JSON:"""

        try:
            response = await self.llm_service.generate_response(prompt)
            if not response or not response.strip():
                logger.warning("LLM concept extraction returned empty response")
                return {}
            
            # Clean response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM concept extraction failed - invalid JSON: {e}. Response: {response[:200] if response else 'Empty'}")
            return {}
        except Exception as e:
            logger.warning(f"LLM concept extraction failed: {e}")
            return {}

    async def _assess_quality_with_llm(self, content: str) -> Dict:
        """Use LLM to assess content quality and complexity."""

        prompt = f"""Assess the quality and characteristics of this content. Return JSON with:

readability_score: Score from 0.0 to 1.0 for how readable the content is
complexity_indicators: Array of factors that make this content complex or simple
quality_indicators: Object with boolean fields: {{"well_structured": true, "has_examples": true, "clear_explanations": true, "good_formatting": true}}
estimated_reading_time: Estimated minutes to read and understand this content

Content to assess:
{content[:1500]}...

Return only valid JSON:"""

        try:
            response = await self.llm_service.generate_response(prompt)
            if not response or not response.strip():
                logger.warning("LLM quality assessment returned empty response")
                return {"readability_score": 0.8}
            
            # Clean response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM quality assessment failed - invalid JSON: {e}. Response: {response[:200] if response else 'Empty'}")
            return {"readability_score": 0.8}
        except Exception as e:
            logger.warning(f"LLM quality assessment failed: {e}")
            return {"readability_score": 0.8}

    def _combine_llm_results(
        self,
        content: str,
        url: Optional[str],
        classification: Dict,
        code_analysis: Dict,
        concepts: Dict,
        quality: Dict,
    ) -> ContentAnalysis:
        """Combine LLM analysis results into ContentAnalysis object."""

        return ContentAnalysis(
            # Core metadata from LLM
            content_type=classification.get("content_type", "general"),
            primary_language=code_analysis.get("primary_language"),
            summary=classification.get("summary", ""),
            # Code analysis
            code_percentage=float(code_analysis.get("code_percentage", 0)),
            code_blocks=code_analysis.get("code_blocks", []),
            # Patterns and concepts
            detected_patterns={
                "programming_patterns": code_analysis.get("programming_patterns", []),
                "complexity_level": code_analysis.get("complexity_level", "moderate"),
                "technologies": concepts.get("technologies_mentioned", []),
            },
            key_concepts=concepts.get("key_concepts", []),
            api_references=concepts.get("api_references", []),
            # Enhanced fields
            url=url,
            topic_keywords=concepts.get("topic_keywords", []),
            code_elements=self._extract_code_elements_from_blocks(
                code_analysis.get("code_blocks", [])
            ),
            complexity_indicators=quality.get("complexity_indicators", []),
            readability_score=quality.get("readability_score", 0.8),
            quality_indicators=quality.get("quality_indicators", {}),
            raw_content=content[:1000]
            if len(content) > 1000
            else content,  # Store sample for debugging
        )

    async def _enhanced_fallback_analysis(
        self, content: str, url: Optional[str] = None
    ) -> ContentAnalysis:
        """Enhanced fallback analysis that's smarter than pure regex patterns."""

        # Extract code blocks
        code_blocks = self._extract_markdown_code_blocks(content)

        # Improved content type classification
        content_type = self._classify_content_type_enhanced(content, code_blocks)

        # Smart language detection
        primary_language = self._detect_language_enhanced(content, code_blocks)

        # Enhanced concept extraction
        key_concepts = self._extract_concepts_enhanced(content)

        # Smart API reference detection
        api_references = self._extract_api_references_enhanced(content)

        # Calculate metrics
        code_percentage = self._calculate_code_percentage(content, code_blocks)
        summary = self._generate_smart_summary(content, content_type)

        return ContentAnalysis(
            content_type=content_type,
            primary_language=primary_language,
            summary=summary,
            code_percentage=code_percentage,
            code_blocks=code_blocks,
            detected_patterns=self._detect_patterns_enhanced(content),
            key_concepts=key_concepts,
            api_references=api_references,
            url=url,
            topic_keywords=key_concepts,  # Use same as key_concepts for fallback
            code_elements=self._extract_code_elements_from_blocks(code_blocks),
            readability_score=self._estimate_readability(content),
            quality_indicators=self._assess_quality_heuristics(content),
        )

    def _extract_markdown_code_blocks(self, content: str) -> List[CodeBlock]:
        """Enhanced code block extraction with better language detection."""

        code_blocks = []

        # Enhanced regex for code blocks
        patterns = [
            r"```(\w+)?\n(.*?)```",  # Standard markdown
            r"~~~(\w+)?\n(.*?)~~~",  # Alternative markdown
            r"<code[^>]*>(.*?)</code>",  # HTML code tags
            r"`([^`\n]{10,})`",  # Inline code that's long enough to be significant
        ]

        for pattern in patterns:
            logger.debug(f"Processing LLM code block pattern: {pattern}")
            matches = re.finditer(pattern, content, re.DOTALL)

            for match in matches:
                try:
                    # Debug logging for match analysis
                    logger.debug(f"LLM code block match: {match.group(0)[:100]}...")
                    logger.debug(f"LLM code block groups: {match.groups()}")
                    logger.debug(f"LLM code block group types: {[type(g) for g in match.groups()]}")
                    
                    if len(match.groups()) >= 2:
                        language = match.group(1) or "unknown"
                        code_content = match.group(2).strip()
                        logger.debug(f"LLM two groups - language: {repr(language)}, code_content type: {type(code_content)}")
                    else:
                        language = "unknown"
                        code_content = match.group(1).strip()
                        logger.debug(f"LLM one group - code_content type: {type(code_content)}")
                        
                except Exception as e:
                    logger.error(f"Error processing LLM code block match: {e}")
                    logger.error(f"Match object: {match}")
                    logger.error(f"Match groups: {match.groups()}")
                    logger.error(f"Pattern: {pattern}")
                    raise

                # Skip very short snippets
                if len(code_content) < 10:
                    continue

                # Enhance language detection
                if language == "unknown" or not language:
                    language = self._detect_language_from_code(code_content)

                # Calculate positions
                start_pos = match.start()
                lines_before = content[:start_pos].count("\n")
                lines_in_block = code_content.count("\n")

                # Extract code elements using enhanced methods
                try:
                    functions = self._extract_functions_enhanced(code_content, language)
                    classes = self._extract_classes_enhanced(code_content, language)
                    imports = self._extract_imports_enhanced(code_content, language)
                except Exception as e:
                    logger.warning(f"Failed to extract code elements: {e}")
                    functions = []
                    classes = []
                    imports = []

                code_block = CodeBlock(
                    language=language.lower(),
                    content=code_content,
                    start_line=lines_before + 1,
                    end_line=lines_before + lines_in_block + 2,
                    functions=functions,
                    classes=classes,
                    imports=imports,
                )
                code_blocks.append(code_block)

        return code_blocks

    def _detect_language_from_code(self, code: str) -> str:
        """Smart language detection from code content using multiple heuristics."""

        # Language scoring system
        language_scores = {}

        # Python indicators
        python_patterns = [
            (r"\bdef\s+\w+\s*\(", 3),
            (r"\bclass\s+\w+\s*:", 3),
            (r"\bimport\s+\w+", 2),
            (r"\bfrom\s+\w+\s+import", 3),
            (r":\s*$", 1),  # Colon at end of line
            (r"\bself\b", 2),
            (r"\b__\w+__\b", 2),  # Dunder methods
        ]

        # JavaScript/TypeScript indicators
        js_patterns = [
            (r"\bfunction\s+\w+\s*\(", 3),
            (r"\bconst\s+\w+\s*=", 2),
            (r"\blet\s+\w+\s*=", 2),
            (r"=>\s*{", 3),  # Arrow functions
            (r"\bconsole\.log", 2),
            (r"\brequire\s*\(", 2),
            (r"\bexport\s+", 2),
        ]

        # Add scoring for each language
        language_patterns = {
            "python": python_patterns,
            "javascript": js_patterns,
            "typescript": js_patterns + [(r":\s*\w+\s*[=;]", 2)],  # Type annotations
            "java": [
                (r"\bpublic\s+class\s+\w+", 3),
                (r"\bpublic\s+static\s+void\s+main", 3),
                (r"\bprivate\s+\w+", 2),
                (r"\bimport\s+java\.", 2),
            ],
            "go": [
                (r"\bpackage\s+\w+", 3),
                (r"\bfunc\s+\w+\s*\(", 3),
                (r"\bimport\s+\"", 2),
            ],
            "rust": [
                (r"\bfn\s+\w+\s*\(", 3),
                (r"\bstruct\s+\w+", 2),
                (r"\bimpl\s+\w+", 2),
                (r"\buse\s+\w+", 2),
            ],
        }

        # Score each language
        for language, patterns in language_patterns.items():
            score = 0
            for pattern, weight in patterns:
                matches = len(re.findall(pattern, code, re.MULTILINE))
                score += matches * weight
            language_scores[language] = score

        # Return highest scoring language
        if language_scores and max(language_scores.values()) > 0:
            return max(language_scores, key=language_scores.get)

        return "unknown"

    def _classify_content_type_enhanced(
        self, content: str, code_blocks: List[CodeBlock]
    ) -> str:
        """Enhanced content type classification using multiple signals."""

        content_lower = content.lower()

        # Calculate code percentage
        code_percentage = self._calculate_code_percentage(content, code_blocks)

        # Multi-signal classification
        signals = {
            "tutorial": 0,
            "api_reference": 0,
            "code_example": 0,
            "troubleshooting": 0,
            "concept_explanation": 0,
        }

        # Tutorial signals
        tutorial_indicators = [
            ("tutorial", 3),
            ("guide", 2),
            ("walkthrough", 3),
            ("step by step", 3),
            ("getting started", 3),
            ("how to", 2),
            ("learn", 1),
            ("introduction", 2),
        ]

        for indicator, weight in tutorial_indicators:
            if indicator in content_lower:
                signals["tutorial"] += weight

        # API reference signals
        api_indicators = [
            ("api", 2),
            ("reference", 2),
            ("documentation", 1),
            ("parameters", 3),
            ("returns", 2),
            ("endpoint", 3),
            ("method", 1),
            ("function", 1),
        ]

        for indicator, weight in api_indicators:
            if indicator in content_lower:
                signals["api_reference"] += weight

        # Code example signals
        if code_percentage > 30:
            signals["code_example"] += 5
        if code_percentage > 50:
            signals["code_example"] += 10

        example_indicators = [("example", 2), ("sample", 2), ("demo", 2)]
        for indicator, weight in example_indicators:
            if indicator in content_lower:
                signals["code_example"] += weight

        # Troubleshooting signals
        trouble_indicators = [
            ("error", 2),
            ("problem", 2),
            ("issue", 2),
            ("troubleshoot", 3),
            ("debug", 2),
            ("fix", 1),
            ("solution", 2),
            ("resolve", 2),
        ]

        for indicator, weight in trouble_indicators:
            if indicator in content_lower:
                signals["troubleshooting"] += weight

        # Concept explanation signals
        concept_indicators = [
            ("concept", 2),
            ("theory", 2),
            ("principle", 2),
            ("overview", 1),
            ("architecture", 2),
            ("design", 1),
            ("pattern", 1),
            ("explanation", 2),
        ]

        for indicator, weight in concept_indicators:
            if indicator in content_lower:
                signals["concept_explanation"] += weight

        # Return highest scoring type
        max_score = max(signals.values())
        if max_score > 0:
            return max(signals, key=signals.get)

        return "general"

    def _detect_language_enhanced(
        self, content: str, code_blocks: List[CodeBlock]
    ) -> Optional[str]:
        """Enhanced language detection using code blocks and content analysis."""

        # Priority 1: Explicit code blocks
        if code_blocks:
            languages = [
                block.language for block in code_blocks if block.language != "unknown"
            ]
            if languages:
                from collections import Counter

                return Counter(languages).most_common(1)[0][0]

        # Priority 2: Content analysis
        content_lower = content.lower()

        # File extension mentions
        extension_patterns = {
            "python": [r"\.py\b", r"python"],
            "javascript": [r"\.js\b", r"javascript", r"node\.js"],
            "typescript": [r"\.ts\b", r"typescript"],
            "java": [r"\.java\b", r"\bjava\b"],
            "go": [r"\.go\b", r"\bgolang\b"],
            "rust": [r"\.rs\b", r"\brust\b"],
            "c": [r"\.c\b(?!\w)", r"\bc\s+language"],
            "cpp": [r"\.cpp\b", r"c\+\+"],
        }

        for language, patterns in extension_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return language

        return None

    def _extract_concepts_enhanced(self, content: str) -> List[str]:
        """Enhanced concept extraction using NLP-like techniques."""

        concepts = set()

        # Extract capitalized terms (likely concepts)
        capitalized = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", content)

        # Filter out common words
        stopwords = {
            "The",
            "This",
            "That",
            "With",
            "From",
            "When",
            "Where",
            "What",
            "How",
            "Why",
            "You",
            "Your",
            "Code",
            "Example",
            "Here",
            "There",
            "Then",
            "Now",
            "Can",
            "Will",
            "Should",
            "Would",
            "Could",
            "May",
            "Might",
            "Must",
            "Need",
            "Get",
            "Set",
            "Put",
            "Post",
            "Delete",
            "Create",
            "Update",
        }

        concepts.update([word for word in capitalized if word not in stopwords])

        # Extract technical terms (CamelCase, snake_case, etc.)
        technical_terms = re.findall(r"\b[a-z]+[A-Z][a-zA-Z]*\b", content)  # CamelCase
        concepts.update(technical_terms)

        # Extract quoted terms (often important concepts)
        quoted = re.findall(r'"([^"]{3,30})"', content)
        quoted.extend(re.findall(r"'([^']{3,30})'", content))
        concepts.update([q for q in quoted if len(q.split()) <= 3])

        # Extract terms from headers
        headers = re.findall(r"#+\s*([^\n]+)", content)
        for header in headers:
            words = re.findall(r"\b[A-Za-z]{3,}\b", header)
            concepts.update(words)

        # Convert to list and rank by frequency
        concept_list = list(concepts)

        # Simple frequency-based ranking
        from collections import Counter

        all_words = re.findall(r"\b[A-Za-z]{3,}\b", content.lower())
        word_counts = Counter(all_words)

        # Score concepts by frequency and length
        scored_concepts = []
        for concept in concept_list:
            freq = word_counts.get(concept.lower(), 0)
            length_bonus = len(concept) / 10  # Prefer longer terms
            score = freq + length_bonus
            scored_concepts.append((concept, score))

        # Return top concepts
        scored_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, score in scored_concepts[:15]]

    def _extract_api_references_enhanced(self, content: str) -> List[str]:
        """Enhanced API reference extraction with context awareness."""

        api_refs = set()

        # Enhanced patterns for API references
        patterns = [
            r"\b(\w+\.\w+)\s*\(",  # Method calls
            r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]+)",  # HTTP endpoints
            r"https?://[^\s]+/api/([^\s]+)",  # API URLs
            r"@(\w+)",  # Decorators/annotations
            r"/api/v?\d*/(\w+)",  # Versioned API endpoints
            r"\b(\w+API|\w+Service|\w+Client)\b",  # API classes
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    api_refs.update([m for m in match if m and len(m) > 2])
                else:
                    if len(match) > 2:
                        api_refs.add(match)

        # Clean and validate API references
        cleaned_refs = []
        for ref in api_refs:
            ref = ref.strip()
            if (
                len(ref) > 2
                and not ref.lower() in ["get", "post", "put", "delete", "patch"]
                and len(ref) < 50
            ):  # Reasonable length limit
                cleaned_refs.append(ref)

        return cleaned_refs[:15]  # Return top 15

    def _detect_patterns_enhanced(self, content: str) -> Dict[str, List[str]]:
        """Enhanced pattern detection with better categorization."""

        patterns = {
            "programming_concepts": [],
            "technologies": [],
            "design_patterns": [],
            "best_practices": [],
        }

        content_lower = content.lower()

        # Programming concepts
        prog_concepts = [
            "async",
            "await",
            "promise",
            "callback",
            "closure",
            "recursion",
            "inheritance",
            "polymorphism",
            "encapsulation",
            "abstraction",
            "concurrency",
            "parallelism",
            "thread",
            "process",
            "synchronization",
        ]

        for concept in prog_concepts:
            if concept in content_lower:
                patterns["programming_concepts"].append(concept)

        # Technologies and frameworks
        technologies = [
            "react",
            "angular",
            "vue",
            "node",
            "express",
            "django",
            "flask",
            "spring",
            "rails",
            "laravel",
            "docker",
            "kubernetes",
            "aws",
            "gcp",
            "azure",
            "mongodb",
            "postgresql",
            "mysql",
            "redis",
            "elasticsearch",
        ]

        for tech in technologies:
            if tech in content_lower:
                patterns["technologies"].append(tech)

        # Design patterns
        design_patterns = [
            "singleton",
            "factory",
            "observer",
            "decorator",
            "adapter",
            "strategy",
            "command",
            "state",
            "proxy",
            "facade",
            "builder",
        ]

        for pattern in design_patterns:
            if pattern in content_lower:
                patterns["design_patterns"].append(pattern)

        # Best practices indicators
        best_practices = [
            "testing",
            "unit test",
            "integration test",
            "code review",
            "documentation",
            "error handling",
            "logging",
            "monitoring",
            "security",
            "performance",
            "optimization",
            "caching",
        ]

        for practice in best_practices:
            if practice in content_lower:
                patterns["best_practices"].append(practice)

        # Remove empty categories
        return {k: v for k, v in patterns.items() if v}

    def _calculate_code_percentage(
        self, content: str, code_blocks: List[CodeBlock]
    ) -> float:
        """Calculate percentage of content that is code."""
        if not content.strip():
            return 0.0

        total_chars = len(content)
        code_chars = sum(len(block.content) for block in code_blocks)

        return (code_chars / total_chars) * 100 if total_chars > 0 else 0.0

    def _generate_smart_summary(self, content: str, content_type: str) -> str:
        """Generate intelligent summary based on content structure."""

        lines = content.split("\n")

        # Extract title from headers
        title = None
        for line in lines[:10]:
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                break

        # Extract key sentences from first few paragraphs
        paragraphs = content.split("\n\n")
        key_sentences = []

        for para in paragraphs[:3]:
            para = para.strip()
            if para and not para.startswith("#") and not para.startswith("```"):
                # Take first sentence of paragraph
                sentences = para.split(". ")
                if sentences:
                    key_sentences.append(sentences[0])

        # Build smart summary
        if title:
            summary = title
            if key_sentences and len(summary) < 100:
                summary += ": " + key_sentences[0][:150]
        elif key_sentences:
            summary = key_sentences[0][:200]
        else:
            summary = f"Content analysis for {content_type}"

        return summary[:300]

    def _estimate_readability(self, content: str) -> float:
        """Estimate readability score using simple heuristics."""

        sentences = len(re.findall(r"[.!?]+", content))
        words = len(content.split())

        if sentences == 0 or words == 0:
            return 0.5

        avg_sentence_length = words / sentences

        # Readability factors
        readability = 1.0

        # Penalty for very long sentences
        if avg_sentence_length > 25:
            readability -= 0.2
        elif avg_sentence_length > 20:
            readability -= 0.1

        # Bonus for moderate sentence length
        if 10 <= avg_sentence_length <= 18:
            readability += 0.1

        # Check for formatting elements (lists, headers, etc.)
        if re.search(r"^\s*[-*]\s", content, re.MULTILINE):  # Lists
            readability += 0.1
        if re.search(r"^#+\s", content, re.MULTILINE):  # Headers
            readability += 0.1

        return max(0.1, min(1.0, readability))

    def _assess_quality_heuristics(self, content: str) -> Dict[str, bool]:
        """Assess content quality using heuristic rules."""

        return {
            "well_structured": bool(re.search(r"^#+\s", content, re.MULTILINE)),
            "has_examples": "example" in content.lower() or "```" in content,
            "clear_explanations": len(content.split()) > 50,  # Sufficient detail
            "good_formatting": bool(re.search(r"^\s*[-*]\s", content, re.MULTILINE)),
        }

    def _extract_code_elements_from_blocks(
        self, code_blocks: List[CodeBlock]
    ) -> List[str]:
        """Extract all code elements from code blocks."""

        elements = []
        for block in code_blocks:
            elements.extend(block.functions)
            elements.extend(block.classes)
            elements.extend(block.imports)

        return list(set(elements))  # Remove duplicates

    def _extract_functions_enhanced(self, code: str, language: str) -> List[str]:
        """Enhanced function extraction with better patterns."""

        functions = []
        language = language.lower()

        patterns = {
            "python": [r"\bdef\s+(\w+)\s*\(", r"\basync\s+def\s+(\w+)\s*\("],
            "javascript": [
                r"\bfunction\s+(\w+)\s*\(",
                r"\b(\w+)\s*=\s*function\s*\(",
                r"\b(\w+)\s*=\s*\([^)]*\)\s*=>",
                r"\bconst\s+(\w+)\s*=\s*\([^)]*\)\s*=>",
            ],
            "typescript": [
                r"\bfunction\s+(\w+)\s*\(",
                r"\b(\w+)\s*=\s*\([^)]*\)\s*=>",
                r"\bconst\s+(\w+)\s*=\s*\([^)]*\)\s*=>",
            ],
            "java": [
                r"\b(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*{"
            ],
            "go": [r"\bfunc\s+(\w+)\s*\("],
            "rust": [r"\bfn\s+(\w+)\s*\("],
            "c": [r"\b(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{"],
            "cpp": [r"\b(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{"],
        }

        if language in patterns:
            for pattern in patterns[language]:
                try:
                    matches = re.findall(pattern, code)
                    # Handle both string and tuple results
                    for match in matches:
                        if isinstance(match, tuple):
                            # Multiple groups - add non-empty ones
                            functions.extend([m for m in match if m and m.strip()])
                        elif isinstance(match, str):
                            # Single group - add if non-empty
                            if match and match.strip():
                                functions.append(match.strip())
                except Exception as e:
                    logger.warning(f"Failed to extract functions with pattern {pattern}: {e}")
                    continue

        return list(set(functions))

    def _extract_classes_enhanced(self, code: str, language: str) -> List[str]:
        """Enhanced class extraction with better patterns."""

        classes = []
        language = language.lower()

        patterns = {
            "python": [r"\bclass\s+(\w+)\s*[:(]"],
            "javascript": [r"\bclass\s+(\w+)\s*[{(]"],
            "typescript": [r"\bclass\s+(\w+)\s*[{(<]", r"\binterface\s+(\w+)\s*{"],
            "java": [r"\b(?:public|private)?\s*(?:abstract\s+)?class\s+(\w+)"],
            "rust": [r"\bstruct\s+(\w+)", r"\benum\s+(\w+)", r"\btrait\s+(\w+)"],
            "c": [r"\bstruct\s+(\w+)", r"\btypedef\s+struct\s+(\w+)"],
            "cpp": [r"\bclass\s+(\w+)", r"\bstruct\s+(\w+)"],
        }

        if language in patterns:
            for pattern in patterns[language]:
                try:
                    matches = re.findall(pattern, code)
                    # Handle both string and tuple results
                    for match in matches:
                        if isinstance(match, tuple):
                            # Multiple groups - add non-empty ones
                            classes.extend([m for m in match if m and m.strip()])
                        elif isinstance(match, str):
                            # Single group - add if non-empty
                            if match and match.strip():
                                classes.append(match.strip())
                except Exception as e:
                    logger.warning(f"Failed to extract classes with pattern {pattern}: {e}")
                    continue

        return list(set(classes))

    def _extract_imports_enhanced(self, code: str, language: str) -> List[str]:
        """Enhanced import extraction with better patterns."""

        imports = []
        language = language.lower()

        patterns = {
            "python": [r"\bimport\s+([\w.]+)", r"\bfrom\s+([\w.]+)\s+import"],
            "javascript": [
                r"\bimport\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
                r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]",
            ],
            "typescript": [
                r"\bimport\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
                r"\bimport\s+['\"]([^'\"]+)['\"]",
            ],
            "java": [r"\bimport\s+([\w.]+)"],
            "go": [r"\bimport\s+[\"']([^\"']+)[\"']"],
            "rust": [r"\buse\s+([\w:]+)"],
            "c": [r"#include\s*[<\"]([^>\"]+)[>\"]"],
            "cpp": [r"#include\s*[<\"]([^>\"]+)[>\"]"],
        }

        if language in patterns:
            for pattern in patterns[language]:
                try:
                    matches = re.findall(pattern, code)
                    # Handle both string and tuple results
                    for match in matches:
                        if isinstance(match, tuple):
                            # Multiple groups - add non-empty ones
                            imports.extend([m for m in match if m and m.strip()])
                        elif isinstance(match, str):
                            # Single group - add if non-empty
                            if match and match.strip():
                                imports.append(match.strip())
                except Exception as e:
                    logger.warning(f"Failed to extract imports with pattern {pattern}: {e}")
                    continue

        return list(set(imports))

    def _format_code_blocks_for_llm(self, code_blocks: List[CodeBlock]) -> str:
        """Format code blocks for LLM analysis."""

        formatted = ""
        for i, block in enumerate(code_blocks[:5]):  # Limit to first 5 blocks
            formatted += f"\n--- Code Block {i+1} ({block.language}) ---\n"
            formatted += block.content[:500]  # Limit length
            if len(block.content) > 500:
                formatted += "\n... (truncated)"
            formatted += "\n"

        return formatted

    def _create_minimal_analysis(self, content: str) -> ContentAnalysis:
        """Create minimal analysis when all methods fail."""

        return ContentAnalysis(
            content_type="unknown",
            primary_language=None,
            summary="Content analysis unavailable",
            code_percentage=0.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
            readability_score=0.5,
            quality_indicators={},
        )
