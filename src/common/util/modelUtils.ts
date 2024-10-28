function getModelFromFile(filePath: string): string {
  const normalizedPath = filePath.replace(/\\/g, '/');
  return normalizedPath.split('/').pop() || '';
}


export function prettyBaseModel(model: string | undefined): string {
  if (!model) return '';
  if (model.endsWith('.bin')) return model.slice(0, -4);
  // [Anthropic]
  if (model.includes('claude-3-opus')) return 'Claude 3 Opus';
  if (model.includes('claude-3-sonnet')) return 'Claude 3 Sonnet';
  // [LM Studio]
  if (model.startsWith('C:\\') || model.startsWith('D:\\'))
    return getModelFromFile(model).replace('.gguf', '');
  // [Ollama]
  if (model.includes(':'))
    return model.replace(':latest', '').replaceAll(':', ' ');
  return model;
}