import streamlit as st
import pandas as pd
import json
import re
from collections import Counter
import io
from datetime import datetime
import hashlib
from typing import Dict, List, Any, Tuple
#python -m streamlit run app.py

# Configure the page
st.set_page_config(
    page_title="AI Essay Grading System",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .feedback-container {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .error-container {
        background-color: #ffeaea;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    
    .success-container {
        background-color: #eafaf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []

class ContentEvaluator:
    """Evaluates essay content using text analysis techniques"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
    
    def evaluate(self, essay_text: str, model_answer: str) -> Tuple[float, str]:
        """Evaluate content relevance and quality"""
        try:
            # Extract keywords from both texts
            essay_keywords = self._extract_keywords(essay_text)
            model_keywords = self._extract_keywords(model_answer)
            
            # Calculate semantic similarity using keyword overlap
            semantic_score = self._calculate_semantic_similarity(essay_keywords, model_keywords)
            
            # Calculate content depth
            depth_score = self._calculate_content_depth(essay_text)
            
            # Calculate keyword density and relevance
            relevance_score = self._calculate_relevance(essay_text, model_answer)
            
            # Combine scores
            content_score = (semantic_score * 0.4 + depth_score * 0.3 + relevance_score * 0.3)
            
            # Generate feedback
            feedback = self._generate_content_feedback(semantic_score, depth_score, relevance_score)
            
            return min(100, max(0, content_score)), feedback
            
        except Exception as e:
            return 50.0, f"Error evaluating content: {str(e)}"
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        return {word for word in words if word not in self.stop_words and len(word) > 2}
    
    def _calculate_semantic_similarity(self, essay_keywords: set, model_keywords: set) -> float:
        """Calculate keyword-based semantic similarity"""
        if not model_keywords:
            return 50.0
        
        overlap = len(essay_keywords.intersection(model_keywords))
        union = len(essay_keywords.union(model_keywords))
        
        if union == 0:
            return 0.0
        
        # Jaccard similarity scaled to 100
        similarity = (overlap / len(model_keywords)) * 100
        return min(100, similarity)
    
    def _calculate_content_depth(self, essay_text: str) -> float:
        """Evaluate content depth based on various factors"""
        sentences = re.split(r'[.!?]+', essay_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = essay_text.split()
        
        # Content depth factors
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words if word.isalpha()))
        vocab_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Length score (300 words = optimal)
        length_score = min(100, (word_count / 300) * 100)
        
        # Complexity score (15-25 words per sentence = optimal)
        if 15 <= avg_sentence_length <= 25:
            complexity_score = 100
        elif 10 <= avg_sentence_length < 15 or 25 < avg_sentence_length <= 30:
            complexity_score = 80
        else:
            complexity_score = 60
        
        # Diversity score
        diversity_score = vocab_diversity * 100
        
        depth_score = (length_score * 0.4 + complexity_score * 0.3 + diversity_score * 0.3)
        return min(100, max(0, depth_score))
    
    def _calculate_relevance(self, essay_text: str, model_answer: str) -> float:
        """Calculate topical relevance"""
        essay_sentences = re.split(r'[.!?]+', essay_text)
        model_sentences = re.split(r'[.!?]+', model_answer)
        
        essay_sentences = [s.strip() for s in essay_sentences if len(s.strip()) > 10]
        model_sentences = [s.strip() for s in model_sentences if len(s.strip()) > 10]
        
        if not essay_sentences or not model_sentences:
            return 50.0
        
        relevance_scores = []
        
        for essay_sent in essay_sentences[:5]:  # Check first 5 sentences
            essay_words = self._extract_keywords(essay_sent)
            max_similarity = 0
            
            for model_sent in model_sentences:
                model_words = self._extract_keywords(model_sent)
                if model_words:
                    overlap = len(essay_words.intersection(model_words))
                    similarity = overlap / len(model_words)
                    max_similarity = max(max_similarity, similarity)
            
            relevance_scores.append(max_similarity)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        return min(100, avg_relevance * 100)
    
    def _generate_content_feedback(self, semantic_score: float, depth_score: float, relevance_score: float) -> str:
        """Generate detailed content feedback"""
        feedback_parts = []
        
        if semantic_score >= 80:
            feedback_parts.append("Excellent alignment with key concepts and terminology.")
        elif semantic_score >= 60:
            feedback_parts.append("Good coverage of important topics with room for more specific terminology.")
        elif semantic_score >= 40:
            feedback_parts.append("Moderate relevance to the topic - include more key terms and concepts.")
        else:
            feedback_parts.append("Limited connection to the main topic - focus more on relevant content.")
        
        if depth_score >= 80:
            feedback_parts.append("Well-developed ideas with good depth and complexity.")
        elif depth_score >= 60:
            feedback_parts.append("Adequate development with opportunity for more detailed explanations.")
        else:
            feedback_parts.append("Ideas need more development and supporting details.")
        
        if relevance_score >= 70:
            feedback_parts.append("Strong topical focus throughout the essay.")
        elif relevance_score >= 50:
            feedback_parts.append("Generally stays on topic with some areas for improvement.")
        else:
            feedback_parts.append("Consider maintaining closer focus on the main topic.")
        
        return " ".join(feedback_parts)


class GrammarEvaluator:
    """Evaluates essay grammar using pattern matching and linguistic rules"""
    
    def evaluate(self, essay_text: str) -> Tuple[float, str]:
        """Evaluate grammar quality of the essay"""
        try:
            # Various grammar checks
            spelling_score = self._check_spelling(essay_text)
            punctuation_score = self._check_punctuation(essay_text)
            sentence_structure_score = self._check_sentence_structure(essay_text)
            capitalization_score = self._check_capitalization(essay_text)
            word_usage_score = self._check_word_usage(essay_text)
            
            # Combine scores
            grammar_score = (
                spelling_score * 0.25 +
                punctuation_score * 0.20 +
                sentence_structure_score * 0.25 +
                capitalization_score * 0.15 +
                word_usage_score * 0.15
            )
            
            # Generate feedback
            feedback = self._generate_grammar_feedback(
                spelling_score, punctuation_score, sentence_structure_score,
                capitalization_score, word_usage_score
            )
            
            return min(100, max(0, grammar_score)), feedback
            
        except Exception as e:
            return 70.0, f"Error evaluating grammar: {str(e)}"
    
    def _check_spelling(self, text: str) -> float:
        """Check for common spelling errors"""
        common_errors = [
            (r'\bteh\b', 'the'), (r'\bhte\b', 'the'), (r'\brecieve\b', 'receive'),
            (r'\bseperate\b', 'separate'), (r'\bdefinately\b', 'definitely'),
            (r'\boccured\b', 'occurred'), (r'\bneccessary\b', 'necessary'),
            (r'\btommorow\b', 'tomorrow'), (r'\baccomodate\b', 'accommodate'),
            (r'\bbeleive\b', 'believe'), (r'\bacheive\b', 'achieve'),
            (r'\bcieling\b', 'ceiling'), (r'\bforiegn\b', 'foreign'),
            (r'\bheight\b', 'height'), (r'\bwierd\b', 'weird')
        ]
        
        word_count = len(text.split())
        error_count = 0
        
        for pattern, _ in common_errors:
            errors = re.findall(pattern, text, re.IGNORECASE)
            error_count += len(errors)
        
        # Check for repeated letters (basic typo detection)
        repeated_errors = re.findall(r'\b\w*([a-z])\1{2,}\w*\b', text, re.IGNORECASE)
        error_count += len(repeated_errors)
        
        if word_count == 0:
            return 50.0
        
        error_rate = error_count / word_count
        return max(0, 100 - (error_rate * 200))  # Penalize heavily for spelling errors
    
    def _check_punctuation(self, text: str) -> float:
        """Check punctuation usage"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 50.0
        
        issues = 0
        total_checks = len(sentences)
        
        # Check sentence endings
        sentence_endings = re.findall(r'[.!?]', text)
        if len(sentence_endings) < len(sentences) * 0.8:  # Allow some flexibility
            issues += 1
        
        # Check for spacing issues
        spacing_errors = [
            r'\s+[,.!?;:]',  # Space before punctuation
            r'[,.!?;:][a-zA-Z]',  # Missing space after punctuation
            r'  +',  # Multiple spaces
            r'\s+$',  # Trailing spaces
        ]
        
        for pattern in spacing_errors:
            if re.search(pattern, text):
                issues += 1
                total_checks += 1
        
        # Check for proper comma usage in lists
        if ',' in text:
            # Basic check for comma splices
            comma_splices = re.findall(r'[a-z]+,\s*[A-Z]', text)
            issues += len(comma_splices)
            total_checks += len(comma_splices) + 1
        
        error_rate = issues / total_checks if total_checks > 0 else 0
        return max(0, 100 - (error_rate * 80))
    
    def _check_sentence_structure(self, text: str) -> float:
        """Evaluate sentence structure and variety"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 50.0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        
        # Check for appropriate sentence length variety
        short_sentences = sum(1 for length in sentence_lengths if length < 8)
        long_sentences = sum(1 for length in sentence_lengths if length > 25)
        very_short = sum(1 for length in sentence_lengths if length < 4)
        
        # Calculate variety score
        length_variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
        variety_score = min(100, length_variance * 2)
        
        # Penalties for poor structure
        penalties = 0
        if very_short > len(sentences) * 0.3:  # Too many very short sentences
            penalties += 20
        if long_sentences > len(sentences) * 0.2:  # Too many very long sentences
            penalties += 15
        if avg_length < 8 or avg_length > 22:  # Poor average length
            penalties += 10
        
        structure_score = variety_score - penalties
        return min(100, max(0, structure_score))
    
    def _check_capitalization(self, text: str) -> float:
        """Check capitalization rules"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 50.0
        
        issues = 0
        total_checks = len(sentences)
        
        for sentence in sentences:
            # Check sentence capitalization
            if sentence and not sentence[0].isupper():
                issues += 1
            
            # Check for 'I' capitalization
            i_errors = re.findall(r'\bi\b', sentence)
            issues += len(i_errors)
            total_checks += len(i_errors)
        
        # Check for proper noun capitalization (basic patterns)
        proper_noun_patterns = [
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(america|england|france|germany|china|japan|canada)\b'
        ]
        
        for pattern in proper_noun_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.islower():
                    issues += 1
                total_checks += 1
        
        error_rate = issues / total_checks if total_checks > 0 else 0
        return max(0, 100 - (error_rate * 60))
    
    def _check_word_usage(self, text: str) -> float:
        """Check for word usage issues"""
        words = re.findall(r'\b\w+\b', text.lower())
        words = [word for word in words if len(word) > 2]
        
        if len(words) < 10:
            return 50.0
        
        word_counts = Counter(words)
        total_words = len(words)
        
        # Check for excessive repetition
        repetition_penalty = 0
        for word, count in word_counts.items():
            if count > 3 and len(word) > 3:
                repetition_rate = count / total_words
                if repetition_rate > 0.03:  # More than 3%
                    repetition_penalty += repetition_rate * 100
        
        # Check vocabulary diversity
        unique_words = len(word_counts)
        diversity_ratio = unique_words / total_words
        diversity_score = min(100, diversity_ratio * 150)
        
        usage_score = diversity_score - repetition_penalty
        return min(100, max(0, usage_score))
    
    def _generate_grammar_feedback(self, spelling_score: float, punctuation_score: float,
                                  sentence_structure_score: float, capitalization_score: float,
                                  word_usage_score: float) -> str:
        """Generate detailed grammar feedback"""
        feedback_parts = []
        
        if spelling_score < 70:
            feedback_parts.append("Several spelling errors detected - please review carefully.")
        elif spelling_score < 85:
            feedback_parts.append("Minor spelling issues present.")
        
        if punctuation_score < 70:
            feedback_parts.append("Improve punctuation usage and spacing.")
        elif punctuation_score < 85:
            feedback_parts.append("Minor punctuation improvements needed.")
        
        if sentence_structure_score < 70:
            feedback_parts.append("Vary sentence length and structure more effectively.")
        elif sentence_structure_score < 85:
            feedback_parts.append("Good sentence variety with room for improvement.")
        
        if capitalization_score < 75:
            feedback_parts.append("Check capitalization rules for sentences and proper nouns.")
        
        if word_usage_score < 70:
            feedback_parts.append("Reduce word repetition and expand vocabulary.")
        elif word_usage_score < 85:
            feedback_parts.append("Good vocabulary usage overall.")
        
        if not feedback_parts:
            feedback_parts.append("Excellent grammar and mechanics throughout!")
        
        return " ".join(feedback_parts)


class StructureEvaluator:
    """Evaluates essay structure and organization"""
    
    def evaluate(self, essay_text: str) -> Tuple[float, str]:
        """Evaluate essay structure and organization"""
        try:
            # Various structure checks
            length_score = self._check_length(essay_text)
            paragraph_score = self._check_paragraph_structure(essay_text)
            coherence_score = self._check_coherence(essay_text)
            intro_conclusion_score = self._check_intro_conclusion(essay_text)
            transition_score = self._check_transitions(essay_text)
            
            # Combine scores
            structure_score = (
                length_score * 0.20 +
                paragraph_score * 0.25 +
                coherence_score * 0.25 +
                intro_conclusion_score * 0.20 +
                transition_score * 0.10
            )
            
            # Generate feedback
            feedback = self._generate_structure_feedback(
                length_score, paragraph_score, coherence_score,
                intro_conclusion_score, transition_score
            )
            
            return min(100, max(0, structure_score)), feedback
            
        except Exception as e:
            return 60.0, f"Error evaluating structure: {str(e)}"
    
    def _check_length(self, text: str) -> float:
        """Check essay length appropriateness"""
        words = text.split()
        word_count = len([word for word in words if word.strip()])
        
        # Optimal range: 250-500 words
        if 250 <= word_count <= 500:
            return 100.0
        elif 200 <= word_count < 250 or 500 < word_count <= 600:
            return 85.0
        elif 150 <= word_count < 200 or 600 < word_count <= 750:
            return 70.0
        elif 100 <= word_count < 150 or 750 < word_count <= 1000:
            return 55.0
        else:
            return max(20.0, 100 - abs(word_count - 375) * 0.1)
    
    def _check_paragraph_structure(self, text: str) -> float:
        """Analyze paragraph structure"""
        # Split by double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n|\.\s{2,}', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If no clear breaks, estimate by sentence count
        if len(paragraphs) <= 1:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            estimated_paragraphs = max(1, len(sentences) // 4)
        else:
            estimated_paragraphs = len(paragraphs)
        
        # Score based on paragraph count (3-5 is optimal)
        if 3 <= estimated_paragraphs <= 5:
            paragraph_score = 100.0
        elif estimated_paragraphs == 2 or estimated_paragraphs == 6:
            paragraph_score = 80.0
        elif estimated_paragraphs == 1 or estimated_paragraphs == 7:
            paragraph_score = 60.0
        else:
            paragraph_score = max(30.0, 100 - abs(estimated_paragraphs - 4) * 15)
        
        # Check paragraph balance
        if len(paragraphs) > 1:
            paragraph_lengths = [len(p.split()) for p in paragraphs]
            avg_length = sum(paragraph_lengths) / len(paragraph_lengths)
            
            balance_penalty = 0
            for length in paragraph_lengths:
                if length < avg_length * 0.3 or length > avg_length * 2.5:
                    balance_penalty += 10
            
            paragraph_score -= min(balance_penalty, 30)
        
        return min(100, max(0, paragraph_score))
    
    def _check_coherence(self, text: str) -> float:
        """Check logical flow and coherence"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 50.0
        
        coherence_score = 75.0  # Base score
        
        # Check for transition words and phrases
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'nevertheless', 'thus', 'hence',
            'first', 'second', 'third', 'finally', 'in conclusion',
            'for example', 'for instance', 'such as', 'in contrast',
            'on the other hand', 'similarly', 'likewise', 'although',
            'while', 'despite', 'because', 'since', 'as a result'
        ]
        
        transition_count = 0
        text_lower = text.lower()
        
        for transition in transition_words:
            transition_count += text_lower.count(transition)
        
        # Reward appropriate transition usage
        transition_ratio = transition_count / len(sentences)
        if 0.1 <= transition_ratio <= 0.3:
            coherence_score += 20
        elif 0.05 <= transition_ratio < 0.1:
            coherence_score += 10
        elif transition_ratio > 0.3:
            coherence_score += 5  # Too many transitions
        
        # Check for topic consistency
        words = re.findall(r'\b\w+\b', text.lower())
        content_words = [w for w in words if len(w) > 4]  # Focus on longer words
        word_freq = Counter(content_words)
        
        # Reward recurring themes
        top_words = word_freq.most_common(5)
        if top_words and top_words[0][1] >= 2:
            coherence_score += 10
        
        return min(100, max(0, coherence_score))
    
    def _check_intro_conclusion(self, text: str) -> float:
        """Check for proper introduction and conclusion"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 30.0
        
        score = 40.0  # Base score
        
        # Check introduction
        first_sentence = sentences[0].lower()
        intro_indicators = [
            'in this essay', 'this essay will', 'i will discuss', 'this paper',
            'the purpose of', 'introduction', 'to begin', 'first', 'initially',
            'the topic of', 'in this paper', 'this analysis'
        ]
        
        for indicator in intro_indicators:
            if indicator in first_sentence:
                score += 20
                break
        
        # Check if introduction is substantial
        if len(sentences[0].split()) >= 10:
            score += 10
        
        # Check conclusion
        last_sentence = sentences[-1].lower()
        conclusion_indicators = [
            'in conclusion', 'to conclude', 'in summary', 'finally',
            'therefore', 'thus', 'hence', 'overall', 'in the end',
            'to sum up', 'all in all', 'ultimately'
        ]
        
        for indicator in conclusion_indicators:
            if indicator in last_sentence:
                score += 20
                break
        
        # Check if conclusion is substantial
        if len(sentences[-1].split()) >= 8:
            score += 10
        
        return min(100, max(0, score))
    
    def _check_transitions(self, text: str) -> float:
        """Check for smooth transitions between ideas"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 50.0
        
        transition_words = [
            'first', 'second', 'third', 'next', 'then', 'also', 'furthermore',
            'moreover', 'additionally', 'however', 'but', 'although', 'while',
            'despite', 'in contrast', 'on the other hand', 'similarly',
            'likewise', 'for example', 'for instance', 'such as', 'including',
            'therefore', 'thus', 'consequently', 'as a result', 'hence',
            'meanwhile', 'afterwards', 'subsequently', 'previously'
        ]
        
        transition_count = 0
        text_lower = text.lower()
        
        # Count transitions at sentence beginnings (more meaningful)
        for sentence in sentences:
            sentence_start = sentence.lower()[:50]  # Check first 50 characters
            for transition in transition_words:
                if sentence_start.startswith(transition + ' ') or sentence_start.startswith(transition + ','):
                    transition_count += 1
                    break
        
        # Score based on transition density
        transition_ratio = transition_count / len(sentences)
        
        if 0.15 <= transition_ratio <= 0.4:
            return 100.0
        elif 0.1 <= transition_ratio < 0.15 or 0.4 < transition_ratio <= 0.5:
            return 85.0
        elif 0.05 <= transition_ratio < 0.1 or 0.5 < transition_ratio <= 0.6:
            return 70.0
        elif transition_ratio < 0.05:
            return 50.0
        else:
            return 40.0  # Too many transitions
    
    def _generate_structure_feedback(self, length_score: float, paragraph_score: float,
                                   coherence_score: float, intro_conclusion_score: float,
                                   transition_score: float) -> str:
        """Generate detailed structure feedback"""
        feedback_parts = []
        
        if length_score < 70:
            feedback_parts.append("Consider adjusting essay length for better impact.")
        elif length_score < 85:
            feedback_parts.append("Essay length is acceptable but could be optimized.")
        
        if paragraph_score < 70:
            feedback_parts.append("Improve paragraph organization and balance.")
        elif paragraph_score < 85:
            feedback_parts.append("Good paragraph structure with minor improvements needed.")
        
        if coherence_score < 70:
            feedback_parts.append("Strengthen logical flow and idea connections.")
        elif coherence_score < 85:
            feedback_parts.append("Good coherence with room for enhancement.")
        
        if intro_conclusion_score < 70:
            feedback_parts.append("Develop stronger introduction and conclusion sections.")
        elif intro_conclusion_score < 85:
            feedback_parts.append("Good opening and closing with minor enhancements possible.")
        
        if transition_score < 70:
            feedback_parts.append("Add more transitional phrases for smoother flow.")
        elif transition_score < 85:
            feedback_parts.append("Good use of transitions overall.")
        
        if not feedback_parts:
            feedback_parts.append("Excellent structure and organization throughout!")
        
        return " ".join(feedback_parts)


class EssayGrader:
    """Main essay grading system that combines all evaluators"""
    
    def __init__(self):
        self.content_evaluator = ContentEvaluator()
        self.grammar_evaluator = GrammarEvaluator()
        self.structure_evaluator = StructureEvaluator()
    
    def grade_essay(self, essay_text: str, model_answer: str, 
                   content_weight: float = 0.4, grammar_weight: float = 0.3, 
                   structure_weight: float = 0.3) -> Dict[str, Any]:
        """Grade an essay comprehensively"""
        try:
            # Validate inputs
            if not essay_text.strip():
                raise ValueError("Essay text cannot be empty")
            
            if not model_answer.strip():
                raise ValueError("Model answer cannot be empty")
            
            # Evaluate each component
            content_score, content_feedback = self.content_evaluator.evaluate(essay_text, model_answer)
            grammar_score, grammar_feedback = self.grammar_evaluator.evaluate(essay_text)
            structure_score, structure_feedback = self.structure_evaluator.evaluate(essay_text)
            
            # Calculate weighted overall score
            overall_score = (
                content_score * content_weight +
                grammar_score * grammar_weight +
                structure_score * structure_weight
            )
            
            # Prepare result
            result = {
                'essay_text': essay_text[:500] + "..." if len(essay_text) > 500 else essay_text,
                'overall_score': round(min(100, max(0, overall_score)), 2),
                'content_score': round(min(100, max(0, content_score)), 2),
                'grammar_score': round(min(100, max(0, grammar_score)), 2),
                'structure_score': round(min(100, max(0, structure_score)), 2),
                'feedback': {
                    'content': content_feedback,
                    'grammar': grammar_feedback,
                    'structure': structure_feedback
                },
                'weights_used': {
                    'content': content_weight,
                    'grammar': grammar_weight,
                    'structure': structure_weight
                },
                'word_count': len(essay_text.split()),
                'graded_at': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f"Error grading essay: {str(e)}",
                'overall_score': 0,
                'content_score': 0,
                'grammar_score': 0,
                'structure_score': 0,
                'feedback': {'error': str(e)},
                'graded_at': datetime.now().isoformat()
            }


class FileProcessor:
    """Handles processing of uploaded files containing essays"""
    
    @staticmethod
    def process_file(uploaded_file) -> Dict[str, str]:
        """Process uploaded file and extract essays"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'txt':
                return FileProcessor._process_txt_file(uploaded_file)
            elif file_extension == 'csv':
                return FileProcessor._process_csv_file(uploaded_file)
            elif file_extension == 'json':
                return FileProcessor._process_json_file(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return {}
    
    @staticmethod
    def _process_txt_file(uploaded_file) -> Dict[str, str]:
        """Process plain text file"""
        try:
            content = uploaded_file.read().decode('utf-8')
            
            # Check for essay separators
            if '---' in content or '===' in content:
                essays = {}
                separator = '---' if '---' in content else '==='
                parts = content.split(separator)
                
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part and len(part) > 50:  # Minimum essay length
                        essays[f"Essay_{i+1}"] = part
                
                return essays
            else:
                # Single essay
                content = content.strip()
                if len(content) > 50:
                    return {"Essay_1": content}
                else:
                    return {}
                
        except Exception as e:
            st.error(f"Error processing TXT file: {str(e)}")
            return {}
    
    @staticmethod
    def _process_csv_file(uploaded_file) -> Dict[str, str]:
        """Process CSV file with essays"""
        try:
            df = pd.read_csv(uploaded_file)
            essays = {}
            
            # Identify columns
            possible_id_columns = ['id', 'essay_id', 'student_id', 'name', 'student_name']
            possible_text_columns = ['essay', 'text', 'content', 'answer', 'response', 'submission']
            
            id_column = None
            text_column = None
            
            # Find ID column
            for col in df.columns:
                if col.lower() in possible_id_columns:
                    id_column = col
                    break
            
            # Find text column
            for col in df.columns:
                if col.lower() in possible_text_columns:
                    text_column = col
                    break
            
            # Fallback to first columns if not found
            if not text_column and len(df.columns) >= 1:
                text_column = df.columns[0] if not id_column else df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            if not id_column and len(df.columns) >= 2:
                id_column = df.columns[0] if text_column != df.columns[0] else None
            
            # Extract essays
            for index, row in df.iterrows():
                if text_column and not pd.isna(row[text_column]):
                    essay_text = str(row[text_column]).strip()
                    
                    if len(essay_text) > 50:  # Minimum essay length
                        if id_column and not pd.isna(row[id_column]):
                            essay_id = str(row[id_column])
                        else:
                            essay_id = f"Essay_{index + 1}"
                        
                        essays[essay_id] = essay_text
            
            return essays
            
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            return {}
    
    @staticmethod
    def _process_json_file(uploaded_file) -> Dict[str, str]:
        """Process JSON file with essays"""
        try:
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            essays = {}
            
            if isinstance(data, dict):
                # Handle different JSON structures
                if 'essays' in data:
                    essay_data = data['essays']
                else:
                    essay_data = data
                
                if isinstance(essay_data, dict):
                    # Dictionary of essays
                    for key, value in essay_data.items():
                        if isinstance(value, str) and len(value.strip()) > 50:
                            essays[str(key)] = value.strip()
                        elif isinstance(value, dict):
                            # Extract text from nested object
                            text = FileProcessor._extract_text_from_dict(value)
                            if text and len(text) > 50:
                                essays[str(key)] = text
                
                elif isinstance(essay_data, list):
                    # List of essays
                    for i, item in enumerate(essay_data):
                        if isinstance(item, str) and len(item.strip()) > 50:
                            essays[f"Essay_{i+1}"] = item.strip()
                        elif isinstance(item, dict):
                            text = FileProcessor._extract_text_from_dict(item)
                            if text and len(text) > 50:
                                # Try to get ID from the object
                                essay_id = None
                                for id_field in ['id', 'essay_id', 'student_id', 'name']:
                                    if id_field in item:
                                        essay_id = str(item[id_field])
                                        break
                                
                                if not essay_id:
                                    essay_id = f"Essay_{i+1}"
                                
                                essays[essay_id] = text
            
            elif isinstance(data, list):
                # Direct list of essays
                for i, item in enumerate(data):
                    if isinstance(item, str) and len(item.strip()) > 50:
                        essays[f"Essay_{i+1}"] = item.strip()
                    elif isinstance(item, dict):
                        text = FileProcessor._extract_text_from_dict(item)
                        if text and len(text) > 50:
                            essays[f"Essay_{i+1}"] = text
            
            return essays
            
        except Exception as e:
            st.error(f"Error processing JSON file: {str(e)}")
            return {}
    
    @staticmethod
    def _extract_text_from_dict(data_dict: Dict[str, Any]) -> str:
        """Extract text content from dictionary"""
        text_fields = ['text', 'content', 'essay', 'answer', 'response', 'body', 'submission']
        
        for field in text_fields:
            if field in data_dict and isinstance(data_dict[field], str):
                return data_dict[field].strip()
        
        # Fallback: concatenate all string values that look like essay content
        text_parts = []
        for key, value in data_dict.items():
            if isinstance(value, str) and len(value.strip()) > 30:
                text_parts.append(value.strip())
        
        return ' '.join(text_parts)


def export_results_csv(results: List[Dict[str, Any]]) -> str:
    """Export results to CSV format"""
    try:
        flattened_results = []
        
        for result in results:
            flattened = {
                'essay_id': result.get('essay_id', 'Unknown'),
                'overall_score': result.get('overall_score', 0),
                'content_score': result.get('content_score', 0),
                'grammar_score': result.get('grammar_score', 0),
                'structure_score': result.get('structure_score', 0),
                'word_count': result.get('word_count', 0),
                'content_feedback': result.get('feedback', {}).get('content', ''),
                'grammar_feedback': result.get('feedback', {}).get('grammar', ''),
                'structure_feedback': result.get('feedback', {}).get('structure', ''),
                'graded_at': result.get('graded_at', '')
            }
            
            # Add weights if available
            if 'weights_used' in result:
                for weight_type, weight_value in result['weights_used'].items():
                    flattened[f'{weight_type}_weight'] = weight_value
            
            flattened_results.append(flattened)
        
        df = pd.DataFrame(flattened_results)
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error exporting to CSV: {str(e)}")
        return ""


def export_results_json(results: List[Dict[str, Any]]) -> str:
    """Export results to JSON format"""
    try:
        export_data = {
            'export_info': {
                'total_essays': len(results),
                'export_timestamp': datetime.now().isoformat(),
                'grading_system': 'AI Essay Grader v1.0'
            },
            'results': results
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        st.error(f"Error exporting to JSON: {str(e)}")
        return ""


def calculate_grade_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate grade distribution from results"""
    distribution = {
        'A (90-100)': 0,
        'B (80-89)': 0,
        'C (70-79)': 0,
        'D (60-69)': 0,
        'F (0-59)': 0
    }
    
    for result in results:
        score = result.get('overall_score', 0)
        if score >= 90:
            distribution['A (90-100)'] += 1
        elif score >= 80:
            distribution['B (80-89)'] += 1
        elif score >= 70:
            distribution['C (70-79)'] += 1
        elif score >= 60:
            distribution['D (60-69)'] += 1
        else:
            distribution['F (0-59)'] += 1
    
    return distribution


def main():
    """Main application function"""
    
    # Header
    st.title("🤖 AI-Powered Essay Grading System")
    st.markdown("""
    **Advanced automated essay evaluation using sophisticated text analysis**
    
    This system evaluates essays across three key dimensions:
    - **Content**: Relevance, depth, and alignment with reference materials
    - **Grammar**: Spelling, punctuation, sentence structure, and mechanics  
    - **Structure**: Organization, coherence, transitions, and essay flow
    """)
    
    # Initialize grader
    if 'grader' not in st.session_state:
        with st.spinner("Initializing AI grading system..."):
            st.session_state.grader = EssayGrader()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Grading Configuration")
        
        # Model answer input
        st.subheader("📋 Reference Answer")
        model_answer = st.text_area(
            "Enter the model/reference answer for content evaluation:",
            height=150,
            help="This text will be used as the reference for evaluating content relevance and coverage"
        )
        
        # Scoring weights
        st.subheader("⚖️ Scoring Weights")
        st.write("Adjust the importance of each evaluation component:")
        
        content_weight = st.slider("Content Weight", 0.0, 1.0, 0.4, 0.05,
                                  help="Weight for content relevance and depth")
        grammar_weight = st.slider("Grammar Weight", 0.0, 1.0, 0.3, 0.05,
                                  help="Weight for grammar and mechanics")
        structure_weight = st.slider("Structure Weight", 0.0, 1.0, 0.3, 0.05,
                                    help="Weight for organization and coherence")
        
        # Normalize weights
        total_weight = content_weight + grammar_weight + structure_weight
        if total_weight > 0:
            content_weight /= total_weight
            grammar_weight /= total_weight
            structure_weight /= total_weight
            
            st.write("**Normalized weights:**")
            st.write(f"Content: {content_weight:.2f}")
            st.write(f"Grammar: {grammar_weight:.2f}")
            st.write(f"Structure: {structure_weight:.2f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Grade Essays", "📊 View Results", "📈 Analytics", "ℹ️ Help"])
    
    with tab1:
        st.header("📝 Essay Grading")
        
        # Upload method selection
        upload_method = st.radio(
            "**Choose your grading method:**",
            ["✏️ Single Essay (Text Input)", "📄 Upload File", "📁 Batch Upload"],
            help="Select how you want to submit essays for grading"
        )
        
        if upload_method == "✏️ Single Essay (Text Input)":
            st.subheader("Enter Essay Text")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                essay_text = st.text_area(
                    "Paste the essay content below:",
                    height=400,
                    placeholder="Enter the complete essay text here...",
                    help="Copy and paste the essay you want to grade"
                )
            
            with col2:
                if essay_text:
                    word_count = len(essay_text.split())
                    char_count = len(essay_text)
                    
                    st.markdown("**📊 Text Statistics:**")
                    st.metric("Words", word_count)
                    st.metric("Characters", char_count)
                    
                    if word_count < 100:
                        st.warning("⚠️ Essay seems quite short")
                    elif word_count > 1000:
                        st.warning("⚠️ Essay is quite long")
                    else:
                        st.success("✅ Good length")
            
            if st.button("🎯 Grade This Essay", type="primary", use_container_width=True):
                if not essay_text.strip():
                    st.error("❌ Please enter essay text to grade")
                elif not model_answer.strip():
                    st.error("❌ Please provide a reference answer in the sidebar")
                elif len(essay_text.split()) < 20:
                    st.error("❌ Essay is too short for meaningful grading (minimum 20 words)")
                else:
                    with st.spinner("🤖 AI is analyzing the essay..."):
                        result = st.session_state.grader.grade_essay(
                            essay_text, 
                            model_answer,
                            content_weight,
                            grammar_weight,
                            structure_weight
                        )
                        result['essay_id'] = f"Essay_{len(st.session_state.results) + 1}"
                        st.session_state.results.append(result)
                    
                    st.markdown("""
                    <div class="success-container">
                        <h3>✅ Essay Graded Successfully!</h3>
                        <p>The essay has been analyzed and scored. Check the Results tab to view detailed feedback.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show quick results
                    st.subheader("🎯 Quick Results")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Overall Score", f"{result['overall_score']}/100", 
                                help="Weighted average of all components")
                    with col2:
                        st.metric("Content", f"{result['content_score']}/100",
                                help="Relevance and depth of content")
                    with col3:
                        st.metric("Grammar", f"{result['grammar_score']}/100",
                                help="Language mechanics and usage")
                    with col4:
                        st.metric("Structure", f"{result['structure_score']}/100",
                                help="Organization and coherence")
        
        elif upload_method == "📄 Upload File":
            st.subheader("Upload Essay File")
            
            uploaded_file = st.file_uploader(
                "Choose a file containing essays:",
                type=['txt', 'csv', 'json'],
                help="""
                **Supported formats:**
                - **TXT**: Plain text (use '---' to separate multiple essays)
                - **CSV**: Spreadsheet with essay columns
                - **JSON**: Structured data with essay content
                """
            )
            
            if uploaded_file is not None:
                st.success(f"📄 File uploaded: {uploaded_file.name}")
                
                # Show file preview
                if st.checkbox("👀 Preview file content", help="Show a preview of the uploaded file"):
                    try:
                        if uploaded_file.type == "text/plain":
                            content = uploaded_file.read().decode('utf-8')
                            st.text_area("File preview:", content[:1000] + ("..." if len(content) > 1000 else ""), height=200)
                            uploaded_file.seek(0)  # Reset file pointer
                        elif uploaded_file.type == "text/csv":
                            df = pd.read_csv(uploaded_file)
                            st.write("CSV preview:")
                            st.dataframe(df.head())
                            uploaded_file.seek(0)  # Reset file pointer
                    except Exception as e:
                        st.error(f"Error previewing file: {e}")
                
                if st.button("🚀 Process File", type="primary", use_container_width=True):
                    if not model_answer.strip():
                        st.error("❌ Please provide a reference answer in the sidebar")
                    else:
                        essays = FileProcessor.process_file(uploaded_file)
                        
                        if not essays:
                            st.error("❌ No valid essays found in the file")
                        else:
                            st.success(f"✅ Found {len(essays)} essays in the file")
                            
                            # Process essays with progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, (essay_id, essay_text) in enumerate(essays.items()):
                                status_text.text(f"🤖 Grading essay {i+1}/{len(essays)}: {essay_id}")
                                
                                result = st.session_state.grader.grade_essay(
                                    essay_text,
                                    model_answer,
                                    content_weight,
                                    grammar_weight,
                                    structure_weight
                                )
                                result['essay_id'] = essay_id
                                st.session_state.results.append(result)
                                
                                progress_bar.progress((i + 1) / len(essays))
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            st.markdown(f"""
                            <div class="success-container">
                                <h3>🎉 Batch Processing Complete!</h3>
                                <p>Successfully graded {len(essays)} essays. View results in the Results tab.</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        elif upload_method == "📁 Batch Upload":
            st.subheader("Batch Upload Multiple Files")
            
            uploaded_files = st.file_uploader(
                "Choose multiple files:",
                type=['txt', 'csv', 'json'],
                accept_multiple_files=True,
                help="Upload multiple files to process all essays at once"
            )
            
            if uploaded_files:
                st.success(f"📁 {len(uploaded_files)} files uploaded")
                
                # Show file list
                with st.expander("📋 View uploaded files"):
                    for file in uploaded_files:
                        st.write(f"- {file.name} ({file.type})")
                
                if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                    if not model_answer.strip():
                        st.error("❌ Please provide a reference answer in the sidebar")
                    else:
                        all_essays = {}
                        
                        # Collect essays from all files
                        for uploaded_file in uploaded_files:
                            essays = FileProcessor.process_file(uploaded_file)
                            all_essays.update(essays)
                        
                        if not all_essays:
                            st.error("❌ No valid essays found in any of the files")
                        else:
                            st.success(f"✅ Found {len(all_essays)} total essays across all files")
                            
                            # Process all essays
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, (essay_id, essay_text) in enumerate(all_essays.items()):
                                status_text.text(f"🤖 Grading essay {i+1}/{len(all_essays)}: {essay_id}")
                                
                                result = st.session_state.grader.grade_essay(
                                    essay_text,
                                    model_answer,
                                    content_weight,
                                    grammar_weight,
                                    structure_weight
                                )
                                result['essay_id'] = essay_id
                                st.session_state.results.append(result)
                                
                                progress_bar.progress((i + 1) / len(all_essays))
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            st.markdown(f"""
                            <div class="success-container">
                                <h3>🎉 Batch Processing Complete!</h3>
                                <p>Successfully graded {len(all_essays)} essays from {len(uploaded_files)} files.</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📊 Grading Results")
        
        if not st.session_state.results:
            st.markdown("""
            <div class="info-message">
                <h3>📝 No Results Yet</h3>
                <p>Grade some essays first to see results here. Use the "Grade Essays" tab to get started.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Results summary
            st.subheader(f"📋 Results Summary ({len(st.session_state.results)} essays)")
            
            # Create summary table
            results_df = pd.DataFrame(st.session_state.results)
            summary_cols = ['essay_id', 'overall_score', 'content_score', 'grammar_score', 'structure_score', 'word_count']
            available_cols = [col for col in summary_cols if col in results_df.columns]
            
            display_df = results_df[available_cols].round(2)
            st.dataframe(display_df, use_container_width=True, height=300)
            
            # Detailed view
            st.subheader("🔍 Detailed Essay Analysis")
            
            selected_idx = st.selectbox(
                "Select an essay for detailed analysis:",
                options=range(len(st.session_state.results)),
                format_func=lambda x: f"{st.session_state.results[x]['essay_id']} (Score: {st.session_state.results[x]['overall_score']}/100)"
            )
            
            if selected_idx is not None:
                result = st.session_state.results[selected_idx]
                
                # Score metrics
                st.subheader("📊 Scores")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    score = result['overall_score']
                    delta_color = "normal"
                    if score >= 90:
                        delta_color = "normal"
                    elif score >= 80:
                        delta_color = "normal"
                    elif score < 60:
                        delta_color = "inverse"
                    
                    st.metric("Overall Score", f"{score}/100", 
                            help="Weighted average of all components")
                
                with col2:
                    st.metric("Content", f"{result['content_score']}/100",
                            help="Relevance and depth evaluation")
                
                with col3:
                    st.metric("Grammar", f"{result['grammar_score']}/100",
                            help="Language mechanics assessment")
                
                with col4:
                    st.metric("Structure", f"{result['structure_score']}/100",
                            help="Organization and flow analysis")
                
                # Feedback section
                st.subheader("💬 Detailed Feedback")
                
                if 'feedback' in result and result['feedback']:
                    feedback = result['feedback']
                    
                    if 'content' in feedback:
                        st.markdown(f"""
                        <div class="feedback-container">
                            <h4>📝 Content Feedback</h4>
                            <p>{feedback['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if 'grammar' in feedback:
                        st.markdown(f"""
                        <div class="feedback-container">
                            <h4>✏️ Grammar Feedback</h4>
                            <p>{feedback['grammar']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if 'structure' in feedback:
                        st.markdown(f"""
                        <div class="feedback-container">
                            <h4>🏗️ Structure Feedback</h4>
                            <p>{feedback['structure']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Essay text preview
                if 'essay_text' in result:
                    with st.expander("📄 View Essay Text"):
                        st.text_area("Essay content:", result['essay_text'], height=200, disabled=True)
                
                # Additional information
                if 'word_count' in result or 'graded_at' in result:
                    st.subheader("ℹ️ Additional Information")
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        if 'word_count' in result:
                            st.write(f"**Word Count:** {result['word_count']}")
                    
                    with info_col2:
                        if 'graded_at' in result:
                            graded_time = result['graded_at']
                            st.write(f"**Graded At:** {graded_time[:19].replace('T', ' ')}")
            
            # Export section
            st.subheader("📤 Export Results")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📊 Export as CSV", use_container_width=True):
                    csv_data = export_results_csv(st.session_state.results)
                    if csv_data:
                        st.download_button(
                            label="⬇️ Download CSV File",
                            data=csv_data,
                            file_name=f"essay_grades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            with export_col2:
                if st.button("📋 Export as JSON", use_container_width=True):
                    json_data = export_results_json(st.session_state.results)
                    if json_data:
                        st.download_button(
                            label="⬇️ Download JSON File",
                            data=json_data,
                            file_name=f"essay_grades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
            
            with export_col3:
                if st.button("🗑️ Clear All Results", type="secondary", use_container_width=True):
                    if st.button("⚠️ Confirm Clear", type="secondary", help="This action cannot be undone"):
                        st.session_state.results = []
                        st.rerun()
    
    with tab3:
        st.header("📈 Analytics Dashboard")
        
        if not st.session_state.results:
            st.markdown("""
            <div class="info-message">
                <h3>📊 No Data Available</h3>
                <p>Grade some essays first to see analytics. The dashboard will show performance trends, score distributions, and insights.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            results_df = pd.DataFrame(st.session_state.results)
            
            # Overall statistics
            st.subheader("📊 Overall Statistics")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Total Essays", len(results_df))
            
            with stat_col2:
                avg_score = results_df['overall_score'].mean()
                st.metric("Average Score", f"{avg_score:.1f}")
            
            with stat_col3:
                highest_score = results_df['overall_score'].max()
                st.metric("Highest Score", f"{highest_score:.1f}")
            
            with stat_col4:
                lowest_score = results_df['overall_score'].min()
                st.metric("Lowest Score", f"{lowest_score:.1f}")
            
            # Score distribution
            st.subheader("📊 Score Distribution")
            
            # Grade distribution
            grade_dist = calculate_grade_distribution(st.session_state.results)
            
            dist_col1, dist_col2 = st.columns(2)
            
            with dist_col1:
                st.write("**Grade Distribution:**")
                for grade, count in grade_dist.items():
                    percentage = (count / len(st.session_state.results)) * 100
                    st.write(f"{grade}: {count} essays ({percentage:.1f}%)")
            
            with dist_col2:
                st.bar_chart(grade_dist)
            
            # Component analysis
            st.subheader("🔍 Component Analysis")
            
            component_col1, component_col2 = st.columns(2)
            
            with component_col1:
                st.write("**Average Scores by Component:**")
                component_averages = {
                    'Content': results_df['content_score'].mean(),
                    'Grammar': results_df['grammar_score'].mean(),
                    'Structure': results_df['structure_score'].mean(),
                    'Overall': results_df['overall_score'].mean()
                }
                
                for component, avg_score in component_averages.items():
                    st.write(f"{component}: {avg_score:.1f}/100")
            
            with component_col2:
                st.bar_chart(component_averages)
            
            # Score trends
            if len(results_df) > 1:
                st.subheader("📈 Score Trends")
                
                # Create a simple trend chart
                results_df_indexed = results_df.reset_index()
                trend_data = results_df_indexed[['overall_score', 'content_score', 'grammar_score', 'structure_score']]
                st.line_chart(trend_data)
            
            # Performance insights
            st.subheader("🎯 Performance Insights")
            
            insights = []
            
            # Content performance
            content_avg = results_df['content_score'].mean()
            if content_avg >= 80:
                insights.append("✅ **Strong Content Performance**: Essays show good alignment with reference materials.")
            elif content_avg < 60:
                insights.append("⚠️ **Content Needs Improvement**: Consider providing clearer guidelines or examples.")
            
            # Grammar performance
            grammar_avg = results_df['grammar_score'].mean()
            if grammar_avg >= 80:
                insights.append("✅ **Good Grammar Standards**: Most essays demonstrate solid language mechanics.")
            elif grammar_avg < 60:
                insights.append("⚠️ **Grammar Focus Needed**: Consider additional grammar instruction or resources.")
            
            # Structure performance
            structure_avg = results_df['structure_score'].mean()
            if structure_avg >= 80:
                insights.append("✅ **Well-Organized Essays**: Students show good understanding of essay structure.")
            elif structure_avg < 60:
                insights.append("⚠️ **Structure Guidance Needed**: Focus on teaching essay organization and transitions.")
            
            # Score consistency
            score_std = results_df['overall_score'].std()
            if score_std > 20:
                insights.append("📊 **High Score Variability**: Large performance gaps suggest diverse skill levels.")
            elif score_std < 10:
                insights.append("📊 **Consistent Performance**: Scores show relatively uniform performance levels.")
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
            # Detailed statistics
            with st.expander("📈 Detailed Statistics"):
                st.write("**Descriptive Statistics:**")
                stats_df = results_df[['overall_score', 'content_score', 'grammar_score', 'structure_score']].describe()
                st.dataframe(stats_df.round(2))
    
    with tab4:
        st.header("ℹ️ Help & Information")
        
        st.subheader("🎯 How It Works")
        st.markdown("""
        The AI Essay Grading System evaluates essays across three key dimensions:
        
        **1. Content Analysis (Default: 40% weight)**
        - Evaluates relevance to the reference answer
        - Assesses content depth and development
        - Analyzes keyword coverage and topic focus
        - Measures semantic similarity with model answer
        
        **2. Grammar & Mechanics (Default: 30% weight)**
        - Checks spelling and common errors
        - Evaluates punctuation usage
        - Analyzes sentence structure and variety
        - Assesses capitalization and word usage
        
        **3. Structure & Organization (Default: 30% weight)**
        - Evaluates essay length appropriateness
        - Analyzes paragraph structure and balance
        - Checks for logical flow and coherence
        - Assesses introduction and conclusion quality
        - Evaluates use of transitions
        """)
        
        st.subheader("📁 File Format Guide")
        
        format_tab1, format_tab2, format_tab3 = st.tabs(["TXT Files", "CSV Files", "JSON Files"])
        
        with format_tab1:
            st.markdown("""
            **Plain Text Files (.txt)**
            
            - Single essay: Just paste the essay text
            - Multiple essays: Separate with `---` on a new line
            
            Example:
            ```
            This is the first essay text...
            
            ---
            
            This is the second essay text...
            ```
            """)
        
        with format_tab2:
            st.markdown("""
            **CSV Files (.csv)**
            
            The system automatically detects columns with these names:
            - **ID columns**: id, essay_id, student_id, name, student_name
            - **Text columns**: essay, text, content, answer, response, submission
            
            Example CSV structure:
            ```
            student_id,essay_text
            Student001,"This is the essay content..."
            Student002,"Another essay goes here..."
            ```
            """)
        
        with format_tab3:
            st.markdown("""
            **JSON Files (.json)**
            
            Supports multiple JSON structures:
            
            1. **Object with essays:**
            ```json
            {
              "essay1": "Essay text here...",
              "essay2": "Another essay..."
            }
            ```
            
            2. **Array of essays:**
            ```json
            [
              "First essay text...",
              "Second essay text..."
            ]
            ```
            
            3. **Structured objects:**
            ```json
            [
              {
                "id": "Student001",
                "text": "Essay content here..."
              }
            ]
            ```
            """)
        
        st.subheader("⚙️ Configuration Tips")
        st.markdown("""
        **Adjusting Weights:**
        - Increase **Content** weight for assignments focusing on subject knowledge
        - Increase **Grammar** weight for language learning assessments
        - Increase **Structure** weight for formal writing assignments
        
        **Reference Answer:**
        - Provide a comprehensive model answer
        - Include key terms and concepts you want students to address
        - The more detailed your reference, the better the content evaluation
        
        **Best Practices:**
        - Essays should be at least 50 words for meaningful analysis
        - Optimal length for analysis: 150-750 words
        - Ensure reference answer covers main topics comprehensively
        """)
        
        st.subheader("🔧 Troubleshooting")
        st.markdown("""
        **Common Issues:**
        
        - **"No essays found"**: Check file format and ensure essays meet minimum length (50 words)
        - **Low content scores**: Verify reference answer is comprehensive and relevant
        - **File upload errors**: Ensure file is properly formatted and not corrupted
        - **Unexpected results**: Check that weights are properly configured
        
        **Performance Notes:**
        - Processing time increases with essay length and quantity
        - Large batch uploads may take several minutes
        - Results are stored in browser session (cleared on refresh)
        """)
        
        st.subheader("📞 Support")
        st.markdown("""
        **System Information:**
        - Version: 1.0.0
        - Last Updated: 2024
        - Platform: Streamlit
        
        For technical support or feature requests, please refer to the system documentation.
        """)

if __name__ == "__main__":
    main()