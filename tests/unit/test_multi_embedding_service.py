"""Unit tests for multi-embedding service and content-aware routing."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from context_server.core.content_analysis import ContentAnalysis
from context_server.core.multi_embedding_service import (
    CohereProvider,
    ContentType,
    EmbeddingModel,
    MultiEmbeddingService,
    OpenAIProvider,
)


class TestOpenAIProvider:
    """Test OpenAI embedding provider."""

    @patch("openai.AsyncOpenAI")
    def test_provider_initialization(self, mock_openai):
        """Test OpenAI provider initialization."""
        provider = OpenAIProvider("text-embedding-3-small", "test-key")

        assert provider.model == "text-embedding-3-small"
        assert provider.api_key == "test-key"
        assert provider.get_dimension() == 1536
        mock_openai.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_embed_text_success(self, mock_openai):
        """Test successful text embedding."""
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAIProvider("text-embedding-3-small", "test-key")
        provider.client = mock_client

        result = await provider.embed_text("test content")

        assert len(result) == 1536
        assert all(x == 0.1 for x in result)
        mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_embed_batch_success(self, mock_openai):
        """Test successful batch embedding."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAIProvider("text-embedding-3-small", "test-key")
        provider.client = mock_client

        result = await provider.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert len(result[0]) == 1536
        assert len(result[1]) == 1536


class TestCohereProvider:
    """Test Cohere embedding provider."""

    def test_provider_initialization(self):
        """Test Cohere provider initialization."""
        provider = CohereProvider("embed-code-v3.0", "test-key")

        assert provider.model == "embed-code-v3.0"
        assert provider.api_key == "test-key"
        assert provider.get_dimension() == 4096

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_embed_text_success(self, mock_client_class):
        """Test successful Cohere text embedding."""
        # Mock httpx client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": {"float": [[0.1] * 4096]}}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        provider = CohereProvider("embed-code-v3.0", "test-key")
        result = await provider.embed_text("test code content")

        assert len(result) == 4096
        assert all(x == 0.1 for x in result)
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_embed_batch_success(self, mock_client_class):
        """Test successful Cohere batch embedding."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "embeddings": {"float": [[0.1] * 4096, [0.2] * 4096]}
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        provider = CohereProvider("embed-code-v3.0", "test-key")
        result = await provider.embed_batch(["code1", "code2"])

        assert len(result) == 2
        assert len(result[0]) == 4096


class TestMultiEmbeddingService:
    """Test multi-embedding service with content-aware routing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = MultiEmbeddingService()

        # Mock providers to avoid API calls
        self.mock_openai_provider = Mock()
        self.mock_openai_provider.get_dimension.return_value = 1536
        self.mock_openai_provider.embed_text = AsyncMock(return_value=[0.1] * 1536)
        self.mock_openai_provider.embed_batch = AsyncMock(
            return_value=[[0.1] * 1536] * 2
        )

        self.mock_cohere_provider = Mock()
        self.mock_cohere_provider.get_dimension.return_value = 4096
        self.mock_cohere_provider.embed_text = AsyncMock(return_value=[0.2] * 4096)
        self.mock_cohere_provider.embed_batch = AsyncMock(
            return_value=[[0.2] * 4096] * 2
        )

        # Replace providers with mocks
        self.service.providers[EmbeddingModel.OPENAI_SMALL] = self.mock_openai_provider
        self.service.providers[EmbeddingModel.VOYAGE_CODE] = self.mock_cohere_provider

    def test_route_content_high_code_percentage(self):
        """Test routing for high code percentage content."""
        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Python code",
            code_percentage=60.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        model = self.service.route_content(analysis)
        assert model == EmbeddingModel.VOYAGE_CODE

    def test_route_content_api_reference(self):
        """Test routing for API reference content."""
        analysis = ContentAnalysis(
            content_type="api_reference",
            primary_language="javascript",
            summary="API documentation",
            code_percentage=20.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        model = self.service.route_content(analysis)
        assert model == EmbeddingModel.VOYAGE_CODE

    def test_route_content_tutorial(self):
        """Test routing for tutorial content."""
        analysis = ContentAnalysis(
            content_type="tutorial",
            primary_language=None,
            summary="Tutorial content",
            code_percentage=5.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        model = self.service.route_content(analysis)
        assert model == EmbeddingModel.OPENAI_SMALL

    def test_route_content_general(self):
        """Test routing for general content."""
        analysis = ContentAnalysis(
            content_type="general",
            primary_language=None,
            summary="General content",
            code_percentage=0.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        model = self.service.route_content(analysis)
        assert model == EmbeddingModel.OPENAI_SMALL

    def test_route_content_by_code_percentage_only(self):
        """Test routing based on code percentage without analysis."""
        # High code percentage should route to code model
        model = self.service.route_content(code_percentage=40.0)
        assert model == EmbeddingModel.VOYAGE_CODE

        # Low code percentage should route to general model
        model = self.service.route_content(code_percentage=5.0)
        assert model == EmbeddingModel.OPENAI_SMALL

    @pytest.mark.asyncio
    async def test_embed_text_with_routing(self):
        """Test text embedding with intelligent routing."""
        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Python code",
            code_percentage=50.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        result = await self.service.embed_text("def test(): pass", analysis)

        assert result["success"] is True
        assert result["model"] == "embed-code-v3.0"  # Cohere code model
        assert result["dimension"] == 4096
        assert len(result["embedding"]) == 4096
        self.mock_cohere_provider.embed_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_text_with_fallback(self):
        """Test text embedding with fallback when primary model fails."""
        # Make Cohere provider fail
        self.mock_cohere_provider.embed_text.side_effect = Exception("API Error")

        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Python code",
            code_percentage=50.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        result = await self.service.embed_text("def test(): pass", analysis)

        assert result["success"] is True
        assert result["model"] == "text-embedding-3-small"  # Fallback model
        assert result["used_fallback"] is True
        assert result["dimension"] == 1536

    @pytest.mark.asyncio
    async def test_embed_text_with_force_model(self):
        """Test text embedding with forced model selection."""
        result = await self.service.embed_text(
            "test content", force_model=EmbeddingModel.OPENAI_SMALL
        )

        assert result["success"] is True
        assert result["model"] == "text-embedding-3-small"
        assert result["dimension"] == 1536
        self.mock_openai_provider.embed_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch_with_routing(self):
        """Test batch embedding with intelligent routing."""
        analyses = [
            ContentAnalysis(
                content_type="code_example",
                primary_language="python",
                summary="Code",
                code_percentage=60.0,
                code_blocks=[],
                detected_patterns={},
                key_concepts=[],
                api_references=[],
            ),
            ContentAnalysis(
                content_type="tutorial",
                primary_language=None,
                summary="Tutorial",
                code_percentage=5.0,
                code_blocks=[],
                detected_patterns={},
                key_concepts=[],
                api_references=[],
            ),
        ]

        results = await self.service.embed_batch(
            ["def test(): pass", "This is a tutorial"], content_analyses=analyses
        )

        assert len(results) == 2

        # First should use Cohere (code)
        assert results[0]["model"] == "embed-code-v3.0"
        assert results[0]["dimension"] == 4096

        # Second should use OpenAI (tutorial)
        assert results[1]["model"] == "text-embedding-3-small"
        assert results[1]["dimension"] == 1536

    @pytest.mark.asyncio
    async def test_embed_batch_with_force_model(self):
        """Test batch embedding with forced model."""
        results = await self.service.embed_batch(
            ["text1", "text2"], force_model=EmbeddingModel.OPENAI_SMALL
        )

        assert len(results) == 2
        assert all(r["model"] == "text-embedding-3-small" for r in results)
        assert all(r["dimension"] == 1536 for r in results)
        self.mock_openai_provider.embed_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_with_multiple_models(self):
        """Test embedding with multiple models for comparison."""
        models = [EmbeddingModel.OPENAI_SMALL, EmbeddingModel.VOYAGE_CODE]

        results = await self.service.embed_with_multiple_models("test content", models)

        assert len(results) == 2
        assert "text-embedding-3-small" in results
        assert "embed-code-v3.0" in results

        # Check OpenAI result
        openai_result = results["text-embedding-3-small"]
        assert openai_result["success"] is True
        assert openai_result["dimension"] == 1536

        # Check Cohere result
        cohere_result = results["embed-code-v3.0"]
        assert cohere_result["success"] is True
        assert cohere_result["dimension"] == 4096

    def test_get_available_models(self):
        """Test getting available models."""
        # Mock API keys
        self.mock_openai_provider.api_key = "test-key"
        self.mock_cohere_provider.api_key = "test-key"

        available = self.service.get_available_models()

        assert "text-embedding-3-small" in available
        assert "embed-code-v3.0" in available

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check for all providers."""
        # Mock API keys
        self.mock_openai_provider.api_key = "test-key"
        self.mock_cohere_provider.api_key = "test-key"

        health = await self.service.health_check()

        assert "text-embedding-3-small" in health
        assert "embed-code-v3.0" in health
        assert health["text-embedding-3-small"] is True
        assert health["embed-code-v3.0"] is True

    @pytest.mark.asyncio
    async def test_health_check_with_failures(self):
        """Test health check when providers fail."""
        # Make providers fail health check
        self.mock_openai_provider.embed_text.side_effect = Exception("API Error")
        self.mock_cohere_provider.embed_text.side_effect = Exception("API Error")
        self.mock_openai_provider.api_key = "test-key"
        self.mock_cohere_provider.api_key = "test-key"

        health = await self.service.health_check()

        assert health["text-embedding-3-small"] is False
        assert health["embed-code-v3.0"] is False

    @pytest.mark.asyncio
    async def test_embed_text_complete_failure(self):
        """Test embedding when all models fail."""
        # Make all providers fail
        self.mock_openai_provider.embed_text.side_effect = Exception("API Error")
        self.mock_cohere_provider.embed_text.side_effect = Exception("API Error")

        analysis = ContentAnalysis(
            content_type="code_example",
            primary_language="python",
            summary="Code",
            code_percentage=50.0,
            code_blocks=[],
            detected_patterns={},
            key_concepts=[],
            api_references=[],
        )

        result = await self.service.embed_text("def test(): pass", analysis)

        assert result["success"] is False
        assert "error" in result
        assert (
            len(result["embedding"]) == 4096
        )  # Dummy embedding with Cohere dimensions
