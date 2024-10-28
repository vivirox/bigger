import { LLM_IF_OAI_Chat, LLM_IF_OAI_Complete, LLM_IF_OAI_Fn, LLM_IF_OAI_Json, LLM_IF_OAI_Vision } from '../../store-llms';

import type { ModelDescriptionSchema } from '../llm.server.types';
import type { OpenAIWire } from './openai.wiretypes';
import { wireGroqModelsListOutputSchema } from './groq.wiretypes';
import { wireMistralModelsListOutputSchema } from './mistral.wiretypes';
import { wireTogetherAIListOutputSchema } from './togetherai.wiretypes';


// [Azure] / [OpenAI]
// https://platform.openai.com/docs/models
const _knownOpenAIChatModels: ManualMappings = [

  // o1-mini
  {
    idPrefix: 'o1-mini',
    label: 'o1 Mini',
    description: 'Supported in Big-AGI 2. Points to the most recent o1-mini snapshot: o1-mini-2024-09-12',
    symLink: 'o1-mini-2024-09-12',
    hidden: true,
    // copied from symlinked
    contextWindow: 128000,
    maxCompletionTokens: 65536,
    trainingDataCutoff: 'Oct 2023',
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Vision],
    pricing: { chatIn: 3, chatOut: 12 },
    isPreview: true,
  },
  {
    hidden: true, // we can't support it in Big-AGI 1
    idPrefix: 'o1-mini-2024-09-12',
    label: 'o1 Mini (2024-09-12)',
    description: 'Supported in Big-AGI 2. Fast, cost-efficient reasoning model tailored to coding, math, and science use cases.',
    contextWindow: 128000,
    maxCompletionTokens: 65536,
    trainingDataCutoff: 'Oct 2023',
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Vision],
    pricing: { chatIn: 3, chatOut: 12 },
    isPreview: true,
  }]

const openAIModelsDenyList: string[] = [
  /* /v1/audio/speech */
  'tts-1-hd', 'tts-1',
  /* /v1/embeddings */
  'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002',
  /* /v1/audio/transcriptions, /v1/audio/translations	*/
  'whisper-1',
  /* /v1/images/generations */
  'dall-e-3', 'dall-e-2',
  /* /v1/completions (Legacy)	 */
  '-turbo-instruct', 'davinci-', 'babbage-',
];

export function openAIModelFilter(model: OpenAIWire.Models.ModelDescription) {
  return !openAIModelsDenyList.some(deny => model.id.includes(deny));
}

export function openAIModelToModelDescription(modelId: string, modelCreated: number, modelUpdated?: number): ModelDescriptionSchema {
  return fromManualMapping(_knownOpenAIChatModels, modelId, modelCreated, modelUpdated);
}

export function azureModelToModelDescription(azureDeploymentRef: string, openAIModelIdBase: string, modelCreated: number, modelUpdated?: number): ModelDescriptionSchema {
  // if the deployment name mataches an OpenAI model prefix, use that
  const known = _knownOpenAIChatModels.find(base => azureDeploymentRef == base.idPrefix);
  return fromManualMapping(_knownOpenAIChatModels, known ? azureDeploymentRef : openAIModelIdBase, modelCreated, modelUpdated);
}


// [Deepseek AI]
const _knownDeepseekChatModels: ManualMappings = [
  // [Models and Pricing](https://platform.deepseek.com/api-docs/pricing)
  // [List Models](https://platform.deepseek.com/api-docs/api/list-models)
  {
    idPrefix: 'deepseek-chat',
    label: 'Deepseek Chat V2',
    description: 'Good at general tasks, 128K context length',
    contextWindow: 128000,
    interfaces: [LLM_IF_OAI_Chat],
    maxCompletionTokens: 4096,
    pricing: {
      chatIn: 0.14,
      chatOut: 0.28,
    },
  },
  {
    idPrefix: 'deepseek-coder',
    label: 'Deepseek Coder V2',
    description: 'Good at coding and math tasks, 128K context length',
    contextWindow: 128000,
    interfaces: [LLM_IF_OAI_Chat],
    maxCompletionTokens: 4096,
    pricing: {
      chatIn: 0.14,
      chatOut: 0.28,
    },
  },
];

export function deepseekModelToModelDescription(deepseekModelId: string): ModelDescriptionSchema {
  return fromManualMapping(_knownDeepseekChatModels, deepseekModelId, undefined, undefined, {
    idPrefix: deepseekModelId,
    label: deepseekModelId.replaceAll(/[_-]/g, ' '),
    description: 'New Deepseek Model',
    contextWindow: 128000,
    maxCompletionTokens: 4096,
    interfaces: [LLM_IF_OAI_Chat], // assume..
    hidden: true,
  });
}


// [LM Studio]
export function lmStudioModelToModelDescription(modelId: string): ModelDescriptionSchema {

  // LM Studio model ID's are the file names of the model files
  function getFileName(filePath: string): string {
    const normalizedPath = filePath.replace(/\\/g, '/');
    return normalizedPath.split('/').pop() || '';
  }

  return fromManualMapping([], modelId, undefined, undefined, {
    idPrefix: modelId,
    label: getFileName(modelId)
      .replace('.gguf', '')
      .replace('.bin', ''),
    // .replaceAll('-', ' '),
    description: `Unknown LM Studio model. File: ${modelId}`,
    contextWindow: null, // 'not provided'
    interfaces: [LLM_IF_OAI_Chat], // assume..
  });
}


// [LocalAI]
const _knownLocalAIChatModels: ManualMappings = [
  {
    idPrefix: 'ggml-gpt4all-j',
    label: 'GPT4All-J',
    description: 'GPT4All-J on LocalAI',
    contextWindow: 2048,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    idPrefix: 'luna-ai-llama2',
    label: 'Luna AI Llama2 Uncensored',
    description: 'Luna AI Llama2 on LocalAI',
    contextWindow: 4096,
    interfaces: [LLM_IF_OAI_Chat],
  },
];

export function localAIModelToModelDescription(modelId: string): ModelDescriptionSchema {
  return fromManualMapping(_knownLocalAIChatModels, modelId, undefined, undefined, {
    idPrefix: modelId,
    label: modelId
      .replace('ggml-', '')
      .replace('.bin', '')
      .replaceAll('-', ' '),
    description: 'Unknown localAI model. Please update `models.data.ts` with this ID',
    contextWindow: null, // 'not provided'
    interfaces: [LLM_IF_OAI_Chat], // assume..
  });
}


// [Mistral]
// updated from the models on: https://docs.mistral.ai/getting-started/models/
// and the pricing available on: https://mistral.ai/technology/#pricing

const _knownMistralChatModels: ManualMappings = [
  // Codestral
  {
    idPrefix: 'codestral-2405',
    label: 'Codestral (2405)',
    description: 'Designed and optimized for code generation tasks.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    pricing: { chatIn: 1, chatOut: 3 },
  },
  {
    idPrefix: 'codestral-latest',
    label: 'Mistral Large (latest)',
    symLink: 'mistral-codestral-2405',
    hidden: true,
    // copied
    description: 'Designed and optimized for code generation tasks.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    pricing: { chatIn: 1, chatOut: 3 },
  },

  // Large
  {
    idPrefix: 'mistral-large-2402',
    label: 'Mistral Large (2402)',
    description: 'Top-tier reasoning for high-complexity tasks.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
    pricing: { chatIn: 4, chatOut: 12 },
    benchmark: { cbaElo: 1159 },
  },
  {
    idPrefix: 'mistral-large-latest',
    label: 'Mistral Large (latest)',
    symLink: 'mistral-large-2402',
    hidden: true,
    // copied
    description: 'Top-tier reasoning for high-complexity tasks.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
    pricing: { chatIn: 4, chatOut: 12 },
    benchmark: { cbaElo: 1159 },
  },

  // Open Mixtral (8x22B)
  {
    idPrefix: 'open-mixtral-8x22b-2404',
    label: 'Open Mixtral 8x22B (2404)',
    description: 'Mixtral 8x22B model',
    contextWindow: 65536,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
    pricing: { chatIn: 2, chatOut: 6 },
  },
  {
    idPrefix: 'open-mixtral-8x22b',
    label: 'Open Mixtral 8x22B',
    symLink: 'open-mixtral-8x22b-2404',
    hidden: true,
    // copied
    description: 'Mixtral 8x22B model',
    contextWindow: 65536,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
    pricing: { chatIn: 2, chatOut: 6 },
  },
  // Medium (Deprecated)
  {
    idPrefix: 'mistral-medium-2312',
    label: 'Mistral Medium (2312)',
    description: 'Ideal for intermediate tasks that require moderate reasoning (Data extraction, Summarizing a Document, Writing emails, Writing a Job Description, or Writing Product Descriptions)',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    pricing: { chatIn: 2.7, chatOut: 8.1 },
    benchmark: { cbaElo: 1148 },
    isLegacy: true,
    hidden: true,
  },
  {
    idPrefix: 'mistral-medium-latest',
    label: 'Mistral Medium (latest)',
    symLink: 'mistral-medium-2312',
    // copied
    description: 'Ideal for intermediate tasks that require moderate reasoning (Data extraction, Summarizing a Document, Writing emails, Writing a Job Description, or Writing Product Descriptions)',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    pricing: { chatIn: 2.7, chatOut: 8.1 },
    benchmark: { cbaElo: 1148 },
    isLegacy: true,
    hidden: true,
  },
  {
    idPrefix: 'mistral-medium',
    label: 'Mistral Medium',
    symLink: 'mistral-medium-2312',
    // copied
    description: 'Ideal for intermediate tasks that require moderate reasoning (Data extraction, Summarizing a Document, Writing emails, Writing a Job Description, or Writing Product Descriptions)',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    pricing: { chatIn: 2.7, chatOut: 8.1 },
    benchmark: { cbaElo: 1148 },
    isLegacy: true,
    hidden: true,
  },

  // Open Mixtral (8x7B) -> currently points to `mistral-small-2312` (as per the docs)
  {
    idPrefix: 'open-mixtral-8x7b',
    label: 'Open Mixtral (8x7B)',
    description: 'A sparse mixture of experts model. As such, it leverages up to 45B parameters but only uses about 12B during inference, leading to better inference throughput at the cost of more vRAM.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    pricing: { chatIn: 0.7, chatOut: 0.7 },
  },
  // Small (deprecated)
  {
    idPrefix: 'mistral-small-2402',
    label: 'Mistral Small (2402)',
    description: 'Suitable for simple tasks that one can do in bulk (Classification, Customer Support, or Text Generation)',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
    pricing: { chatIn: 1, chatOut: 3 },
    hidden: true,
    isLegacy: true,
  },
  {
    idPrefix: 'mistral-small-latest',
    label: 'Mistral Small (latest)',
    symLink: 'mistral-small-2402',
    // copied
    description: 'Suitable for simple tasks that one can do in bulk (Classification, Customer Support, or Text Generation)',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
    pricing: { chatIn: 1, chatOut: 3 },
    hidden: true,
    isLegacy: true,
  },
  {
    idPrefix: 'mistral-small-2312',
    label: 'Mistral Small (2312)',
    description: 'Aka open-mixtral-8x7b. Suitable for simple tasks that one can do in bulk (Classification, Customer Support, or Text Generation)',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    pricing: { chatIn: 1, chatOut: 3 },
    hidden: true,
    isLegacy: true,
  },
  {
    idPrefix: 'mistral-small',
    label: 'Mistral Small',
    symLink: 'mistral-small-2312',
    // copied
    description: 'Aka open-mixtral-8x7b. Suitable for simple tasks that one can do in bulk (Classification, Customer Support, or Text Generation)',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    pricing: { chatIn: 1, chatOut: 3 },
    hidden: true,
    isLegacy: true,
  },


  // Open Mistral (7B) -> currently points to mistral-tiny-2312 (as per the docs)
  {
    idPrefix: 'open-mistral-7b',
    label: 'Open Mistral (7B)',
    description: 'The first dense model released by Mistral AI, perfect for experimentation, customization, and quick iteration. At the time of the release, it matched the capabilities of models up to 30B parameters.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    pricing: { chatIn: 0.25, chatOut: 0.25 },
  },
  // Tiny (deprecated)
  {
    idPrefix: 'mistral-tiny-2312',
    label: 'Mistral Tiny (2312)',
    description: 'Aka open-mistral-7b. Used for large batch processing tasks where cost is a significant factor but reasoning capabilities are not crucial',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    hidden: true,
    isLegacy: true,
  },
  {
    idPrefix: 'mistral-tiny',
    label: 'Mistral Tiny',
    symLink: 'mistral-tiny-2312',
    // copied
    description: 'Aka open-mistral-7b. Used for large batch processing tasks where cost is a significant factor but reasoning capabilities are not crucial',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    hidden: true,
    isLegacy: true,
  },

  {
    idPrefix: 'mistral-embed',
    label: 'Mistral Embed',
    description: 'A model that converts text into numerical vectors of embeddings in 1024 dimensions. Embedding models enable retrieval and retrieval-augmented generation applications.',
    maxCompletionTokens: 1024, // HACK - it's 1024 dimensions, but those are not 'completion tokens'
    contextWindow: 8192, // Updated context window
    interfaces: [],
    pricing: { chatIn: 0.1, chatOut: 0.1 },
    hidden: true,
  },
];

const mistralModelFamilyOrder = [
  'codestral', 'mistral-large', 'open-mixtral-8x22b', 'mistral-medium', 'open-mixtral-8x7b', 'mistral-small', 'open-mistral-7b', 'mistral-tiny', 'mistral-embed', '🔗',
];

export function mistralModelToModelDescription(_model: unknown): ModelDescriptionSchema {
  const model = wireMistralModelsListOutputSchema.parse(_model);
  return fromManualMapping(_knownMistralChatModels, model.id, model.created, undefined, {
    idPrefix: model.id,
    label: model.id.replaceAll(/[_-]/g, ' '),
    description: 'New Mistral Model',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat], // assume..
    hidden: true,
  });
}

export function mistralModelsSort(a: ModelDescriptionSchema, b: ModelDescriptionSchema): number {
  if (a.label.startsWith('🔗') && !b.label.startsWith('🔗')) return 1;
  if (!a.label.startsWith('🔗') && b.label.startsWith('🔗')) return -1;
  const aPrefixIndex = mistralModelFamilyOrder.findIndex(prefix => a.id.startsWith(prefix));
  const bPrefixIndex = mistralModelFamilyOrder.findIndex(prefix => b.id.startsWith(prefix));
  if (aPrefixIndex !== -1 && bPrefixIndex !== -1) {
    if (aPrefixIndex !== bPrefixIndex)
      return aPrefixIndex - bPrefixIndex;
    return b.label.localeCompare(a.label);
  }
  return aPrefixIndex !== -1 ? 1 : -1;
}


// [Together AI]

const _knownTogetherAIChatModels: ManualMappings = [
  {
    idPrefix: 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
    label: 'Nous Hermes 2 - Mixtral 8x7B-DPO',
    description: 'Nous Hermes 2 Mixtral 7bx8 DPO is the new flagship Nous Research model trained over the Mixtral 7bx8 MoE LLM. The model was trained on over 1,000,000 entries of primarily GPT-4 generated data, as well as other high quality data from open datasets across the AI landscape, achieving state of the art performance on a variety of tasks.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    idPrefix: 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT',
    label: 'Nous Hermes 2 - Mixtral 8x7B-SFT',
    description: 'Nous Hermes 2 Mixtral 7bx8 SFT is the new flagship Nous Research model trained over the Mixtral 7bx8 MoE LLM. The model was trained on over 1,000,000 entries of primarily GPT-4 generated data, as well as other high quality data from open datasets across the AI landscape, achieving state of the art performance on a variety of tasks.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    idPrefix: 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    label: 'Mixtral-8x7B Instruct',
    description: 'The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    idPrefix: 'mistralai/Mistral-7B-Instruct-v0.2',
    label: 'Mistral (7B) Instruct v0.2',
    description: 'The Mistral-7B-Instruct-v0.2 Large Language Model (LLM) is an improved instruct fine-tuned version of Mistral-7B-Instruct-v0.1.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    idPrefix: 'NousResearch/Nous-Hermes-2-Yi-34B',
    label: 'Nous Hermes-2 Yi (34B)',
    description: 'Nous Hermes 2 - Yi-34B is a state of the art Yi Fine-tune',
    contextWindow: 4097,
    interfaces: [LLM_IF_OAI_Chat],
  },
] as const;

export function togetherAIModelsToModelDescriptions(wireModels: unknown): ModelDescriptionSchema[] {

  function togetherAIModelToModelDescription(model: { id: string, created: number }) {
    return fromManualMapping(_knownTogetherAIChatModels, model.id, model.created, undefined, {
      idPrefix: model.id,
      label: model.id.replaceAll('/', ' · ').replaceAll(/[_-]/g, ' '),
      description: 'New Togehter AI Model',
      contextWindow: null, // unknown
      interfaces: [LLM_IF_OAI_Chat], // assume..
      hidden: true,
    });
  }

  function togetherAIModelsSort(a: ModelDescriptionSchema, b: ModelDescriptionSchema): number {
    if (a.hidden && !b.hidden)
      return 1;
    if (!a.hidden && b.hidden)
      return -1;
    if (a.created !== b.created)
      return (b.created || 0) - (a.created || 0);
    return a.id.localeCompare(b.id);
  }

  return wireTogetherAIListOutputSchema.parse(wireModels)
    .map(togetherAIModelToModelDescription)
    .sort(togetherAIModelsSort);
}


// Perplexity

const _knownPerplexityChatModels: ModelDescriptionSchema[] = [
  // perplexity models
  {
    id: 'llama-3-sonar-small-32k-chat',
    label: 'Sonar Small Chat',
    description: 'Llama 3 Sonar Small 32k Chat',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    id: 'llama-3-sonar-small-32k-online',
    label: 'Sonar Small Online 🌐',
    description: 'Llama 3 Sonar Small 32k Online',
    contextWindow: 28000,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    id: 'llama-3-sonar-large-32k-chat',
    label: 'Sonar Large Chat',
    description: 'Llama 3 Sonar Large 32k Chat',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    id: 'llama-3-sonar-large-32k-online',
    label: 'Sonar Large Online 🌐',
    description: 'Llama 3 Sonar Large 32k Online',
    contextWindow: 28000,
    interfaces: [LLM_IF_OAI_Chat],
  },

  // opensource models
  {
    id: 'llama-3-8b-instruct',
    label: 'Llama 3 8B Instruct',
    description: 'Llama 3 8B Instruct',
    contextWindow: 8192,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    id: 'llama-3-70b-instruct',
    label: 'Llama 3 70B Instruct',
    description: 'Llama 3 70B Instruct',
    contextWindow: 8192,
    interfaces: [LLM_IF_OAI_Chat],
  },
  {
    id: 'mixtral-8x7b-instruct',
    label: 'Mixtral 8x7B Instruct',
    description: 'Mixtral 8x7B Instruct',
    contextWindow: 16384,
    interfaces: [LLM_IF_OAI_Chat],
  },
];

const perplexityAIModelFamilyOrder = [
  'llama-3-sonar-large', 'llama-3-sonar-small', 'llama-3', 'mixtral', '',
];

export function perplexityAIModelDescriptions() {
  // change this implementation once upstream implements some form of models listing
  return _knownPerplexityChatModels;
}

export function perplexityAIModelSort(a: ModelDescriptionSchema, b: ModelDescriptionSchema): number {
  const aPrefixIndex = perplexityAIModelFamilyOrder.findIndex(prefix => a.id.startsWith(prefix));
  const bPrefixIndex = perplexityAIModelFamilyOrder.findIndex(prefix => b.id.startsWith(prefix));
  // sort by family
  if (aPrefixIndex !== -1 && bPrefixIndex !== -1)
    if (aPrefixIndex !== bPrefixIndex)
      return aPrefixIndex - bPrefixIndex;
  // then by reverse label
  return b.label.localeCompare(a.label);
}


// Groq - https://console.groq.com/docs/models

const _knownGroqModels: ManualMappings = [
  {
    isLatest: true,
    idPrefix: 'llama-3.1-405b-reasoning',
    label: 'Llama 3.1 · 405B',
    description: 'LLaMA 3.1 405B developed by Meta with a context window of 131,072 tokens. Supports tool use.',
    contextWindow: 131072,
    maxCompletionTokens: 8000,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
  },
  {
    isLatest: true,
    idPrefix: 'llama-3.1-70b-versatile',
    label: 'Llama 3.1 · 70B',
    description: 'LLaMA 3.1 70B developed by Meta with a context window of 131,072 tokens. Supports tool use.',
    contextWindow: 131072,
    maxCompletionTokens: 8000,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
  },
  {
    isLatest: true,
    idPrefix: 'llama-3.1-8b-instant',
    label: 'Llama 3.1 · 8B',
    description: 'LLaMA 3.1 8B developed by Meta with a context window of 131,072 tokens. Supports tool use.',
    contextWindow: 131072,
    maxCompletionTokens: 8000,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
  },
  {
    idPrefix: 'llama3-groq-70b-8192-tool-use-preview',
    label: 'Llama 3 Groq · 70B Tool Use',
    description: 'LLaMA 3 70B Tool Use developed by Groq with a context window of 8,192 tokens. Optimized for tool use.',
    contextWindow: 8192,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
  },
  {
    idPrefix: 'llama3-groq-8b-8192-tool-use-preview',
    label: 'Llama 3 Groq · 8B Tool Use',
    description: 'LLaMA 3 8B Tool Use developed by Groq with a context window of 8,192 tokens. Optimized for tool use.',
    contextWindow: 8192,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
  },
  {
    idPrefix: 'llama3-70b-8192',
    label: 'Llama 3 · 70B',
    description: 'LLaMA3 70B developed by Meta with a context window of 8,192 tokens. Supports tool use.',
    contextWindow: 8192,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
    // isLegacy: true,
    hidden: true,
  },
  {
    idPrefix: 'llama3-8b-8192',
    label: 'Llama 3 · 8B',
    description: 'LLaMA3 8B developed by Meta with a context window of 8,192 tokens. Supports tool use.',
    contextWindow: 8192,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
    // isLegacy: true,
    hidden: true,
  },
  {
    idPrefix: 'mixtral-8x7b-32768',
    label: 'Mixtral 8x7B',
    description: 'Mixtral 8x7B developed by Mistral with a context window of 32,768 tokens. Supports tool use.',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
  },
  {
    idPrefix: 'gemma2-9b-it',
    label: 'Gemma 2 · 9B Instruct',
    description: 'Gemma 2 9B developed by Google with a context window of 8,192 tokens. Supports tool use.',
    contextWindow: 8192,
    interfaces: [LLM_IF_OAI_Chat, LLM_IF_OAI_Fn],
  },
  {
    idPrefix: 'gemma-7b-it',
    label: 'Gemma 1.1 · 7B Instruct',
    description: 'Gemma 7B developed by Google with a context window of 8,192 tokens. Supports tool use.',
    contextWindow: 8192,
    interfaces: [LLM_IF_OAI_Chat],
    hidden: true,
  },
];

export function groqModelToModelDescription(_model: unknown): ModelDescriptionSchema {
  const model = wireGroqModelsListOutputSchema.parse(_model);
  return fromManualMapping(_knownGroqModels, model.id, model.created, undefined, {
    idPrefix: model.id,
    label: model.id.replaceAll(/[_-]/g, ' '),
    description: 'New Model',
    contextWindow: 32768,
    interfaces: [LLM_IF_OAI_Chat],
    hidden: true,
  });
}

export function groqModelSortFn(a: ModelDescriptionSchema, b: ModelDescriptionSchema): number {
  // sort hidden at the end
  if (a.hidden && !b.hidden)
    return 1;
  if (!a.hidden && b.hidden)
    return -1;
  // sort as per their order in the known models
  const aIndex = _knownGroqModels.findIndex(base => a.id.startsWith(base.idPrefix));
  const bIndex = _knownGroqModels.findIndex(base => b.id.startsWith(base.idPrefix));
  if (aIndex !== -1 && bIndex !== -1)
    return aIndex - bIndex;
  return a.id.localeCompare(b.id);
}


// Helpers

type ManualMapping = ({
  idPrefix: string,
  isLatest?: boolean,
  isPreview?: boolean,
  isLegacy?: boolean,
  symLink?: string
} & Omit<ModelDescriptionSchema, 'id' | 'created' | 'updated'>);
type ManualMappings = ManualMapping[];

function fromManualMapping(mappings: ManualMappings, id: string, created?: number, updated?: number, fallback?: ManualMapping): ModelDescriptionSchema {

  // find the closest known model, or fall back, or take the last
  const known = mappings.find(base => id === base.idPrefix)
    || mappings.find(base => id.startsWith(base.idPrefix))
    || fallback
    || mappings[mappings.length - 1];

  // label for symlinks
  let label = known.label;
  if (known.symLink && id === known.idPrefix)
    label = `🔗 ${known.label} → ${known.symLink/*.replace(known.idPrefix, '')*/}`;

  // check whether this is a partial map, which indicates an unknown/new variant
  const suffix = id.slice(known.idPrefix.length).trim();

  // return the model description sheet
  return {
    id,
    label: label
      + (suffix ? ` [${suffix.replaceAll('-', ' ').trim()}]` : '')
      + (known.isLatest ? ' 🌟' : '')
      + (known.isLegacy ? /*' 💩'*/ ' [legacy]' : ''),
    created: created || 0,
    updated: updated || created || 0,
    description: known.description,
    contextWindow: known.contextWindow,
    ...(!!known.maxCompletionTokens && { maxCompletionTokens: known.maxCompletionTokens }),
    ...(!!known.trainingDataCutoff && { trainingDataCutoff: known.trainingDataCutoff }),
    interfaces: known.interfaces,
    ...(!!known.benchmark && { benchmark: known.benchmark }),
    ...(!!known.pricing && { pricing: known.pricing }),
    ...(!!known.hidden && { hidden: known.hidden }),
  };
}