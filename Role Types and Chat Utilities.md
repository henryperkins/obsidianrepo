```typescript
import { AI_SENDER } from "@/constants";
import ChainManager from "@/LLMProviders/chainManager";
import { ChatMessage } from "@/sharedState";
import { formatDateTime, getModelKey } from "./utils";

export type Role = "assistant" | "user" | "system";

export const getAIResponse = async (
  userMessage: ChatMessage,
  chainManager: ChainManager,
  addMessage: (message: ChatMessage) => void,
  updateCurrentAiMessage: (message: string) => void,
  updateShouldAbort: (abortController: AbortController | null) => void,
  options: {
    debug?: boolean;
    ignoreSystemMessage?: boolean;
    updateLoading?: (loading: boolean) => void;
    updateLoadingMessage?: (message: string) => void;
  } = {}
) => {
  const abortController = new AbortController();
  updateShouldAbort(abortController);
  try {
    const chatModel = chainManager.chatModelManager.getChatModel();
    const modelConfig = chainManager.chatModelManager.getChatModelConfiguration(getModelKey());
    const memory = chainManager.memoryManager.getMemory();
    const memoryVariables = await memory.loadMemoryVariables({});

    if (modelConfig.streaming) {
      const chain = ChainManager.getChain();
      const chatStream = await chain.stream({
        history: memoryVariables.history,
        input: userMessage.message,
      } as any);

      let fullAIResponse = "";
      for await (const chunk of chatStream) {
        if (abortController.signal.aborted) break;
        const content = typeof chunk === 'string' ? chunk : chunk.content;
        if (typeof content === 'string') {
          fullAIResponse += content;
          updateCurrentAiMessage(fullAIResponse);
        }
      }
      addMessage({
        sender: AI_SENDER,
        message: fullAIResponse,
        isVisible: true,
        timestamp: formatDateTime(new Date()),
      });
    } else {
      const response = await chatModel.call({
        input: userMessage.message,
      });
      addMessage({
        sender: AI_SENDER,
        message: response,
        isVisible: true,
        timestamp: formatDateTime(new Date()),
      });
    }
  } catch (error) {
    console.error("Model request failed:", error);
    let errorMessage = "Model request failed: ";

    if (error instanceof Error) {
      errorMessage += error.message;
      if (error.cause) {
        errorMessage += ` Cause: ${error.cause}`;
      }
    } else if (typeof error === "object" && error !== null) {
      errorMessage += JSON.stringify(error);
    } else {
      errorMessage += String(error);
    }

    addMessage({
      sender: AI_SENDER,
      message: `Error: ${errorMessage}`,
      isVisible: true,
      timestamp: formatDateTime(new Date()),
    });
  }
};

```