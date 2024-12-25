**Add `maxCompletionTokens` to model configuration and settings**

* **src/aiParams.ts**
  - Add `maxCompletionTokens` to `ModelConfig` interface
  - Add `updateModelConfig` function to update model configurations

* **src/settings/model.ts**
  - Add `maxCompletionTokens` to `CopilotSettings` interface

* **src/LLMProviders/chatModelManager.ts**
  - Handle `maxCompletionTokens` in `getModelConfig` method

* **src/LLMProviders/chainManager.ts**
  - Retrieve `maxCompletionTokens` from `modelConfigs` in `createChainWithNewModel`
  - Pass `maxCompletionTokens` to the model constructor in `setChain`

* **src/settings/components/ApiSettings.tsx**
  - Add input fields for `maxCompletionTokens` and `reasoningEffort` in API settings




**Add support for the `o1-preview` model from Azure OpenAI, including `maxCompletionTokens` and `reasoningEffort` settings.**

* **aiParams.ts**
  - Add `maxCompletionTokens` and `reasoningEffort` to `ModelConfig` interface.
  - Implement `updateModelConfig` function to update `modelConfigs` in `settingsAtom` with deep merge using `lodash.merge`.

* **settings/model.ts**
  - Add `modelConfigs: Record<string, ModelConfig>` property to `CopilotSettings`.
  - Initialize `modelConfigs` as an empty object in `DEFAULT_SETTINGS`.

* **ApiSettings.tsx**
  - Conditionally render `ApiSetting` components for `maxCompletionTokens` and `reasoningEffort` when the selected model key starts with "o1".
  - Implement `handleMaxCompletionTokensChange` and `handleReasoningEffortChange` to call `updateModelConfig`.

* **chainManager.ts**
  - Retrieve `maxCompletionTokens` and `reasoningEffort` from `modelConfigs` in `createChainWithNewModel`.
  - Pass `maxCompletionTokens` and `reasoningEffort` to the model constructor in `setChain`.

* **chatModelManager.ts**
  - Update `getModelConfig` to configure `o1-preview` model with `azureOpenAIApiVersion` set to `2024-09-01-preview`.
  - Handle `maxCompletionTokens` and `reasoningEffort` in `handleOpenAIExtraArgs`.

* **README.md**
  - Add section explaining how to configure settings for `o1-preview` deployments.
  - Clarify that the deployment name set in Azure must be used in settings.

To manage Azure OpenAI deployments in src/aiParams.ts:

- 🛠️ Define the `azureOpenAIApiDeployments` array in the `CopilotSettings` interface in src/settings/model.ts.
- 📝 Update the `updateModelConfig` function in src/aiParams.ts to handle Azure-specific configurations.
- 🔄 Use the `updateSetting` function in src/settings/model.ts to update the `azureOpenAIApiDeployments` array.
- 🗂️ Ensure the `azureOpenAIApiDeployments` array is used in the `ApiSettings` component in src/settings/components/ApiSettings.tsx to manage the deployments.
- 🔧 Use the `getModelKey` function in src/aiParams.ts to retrieve the current model key and update the Azure-specific configurations accordingly.

How do I define `azureOpenAIApiDeployments` in `CopilotSettings`?

To define `azureOpenAIApiDeployments` in `CopilotSettings`, follow these steps:

- 🛠️ Add the `azureOpenAIApiDeployments` array to the `CopilotSettings` interface in src/settings/model.ts.
- 📝 Ensure the `azureOpenAIApiDeployments` array is used in the `ApiSettings` component in src/settings/components/ApiSettings.tsx to manage the deployments.
- 🔄 Use the `updateSetting` function in src/settings/model.ts to update the `azureOpenAIApiDeployments` array.
- 🔧 Update the `updateModelConfig` function in src/aiParams.ts to handle Azure-specific configurations.
- 🗂️ Ensure the `azureOpenAIApiDeployments` array is used in the `ApiSettings` component in src/settings/components/ApiSettings.tsx to manage the deployments.
- 🔧 Use the `getModelKey` function in src/aiParams.ts to retrieve the current model key and update the Azure-specific configurations accordingly.
How do I ensure `azureOpenAIApiDeployments` is used in `src/settings/components/ApiSettings.tsx`?

To ensure `azureOpenAIApiDeployments` is used in src/settings/components/ApiSettings.tsx:

- 🛠️ Add a state variable `azureDeployments` to manage the deployments.
- 📝 Create handler functions to add, update, and remove Azure deployments.
- 🔄 Use the `updateSetting` function to update the `azureOpenAIApiDeployments` array.
- 🗂️ Render the Azure deployment settings dynamically based on the `azureDeployments` state.
- 🔧 Add a button to allow users to add new Azure deployments.

How do I use `updateSetting` to update the `azureOpenAIApiDeployments` array?

To use `updateSetting` to update the `azureOpenAIApiDeployments` array, follow these steps:

- 🛠️ Define the `azureOpenAIApiDeployments` array in the `CopilotSettings` interface in src/settings/model.ts.
- 📝 Update the `updateModelConfig` function in src/aiParams.ts to handle Azure-specific configurations.
- 🔄 Use the `updateSetting` function in src/settings/model.ts to update the `azureOpenAIApiDeployments` array.
- 🗂️ Ensure the `azureOpenAIApiDeployments` array is used in the `ApiSettings` component in src/settings/components/ApiSettings.tsx to manage the deployments.
- 🔧 Use the `getModelKey` function in src/aiParams.ts to retrieve the current model key and update the Azure-specific configurations accordingly.

How do I define the `azureOpenAIApiDeployments` array in `src/settings/model.ts`?

To define the `azureOpenAIApiDeployments` array in src/settings/model.ts:

- 🛠️ Add the `azureOpenAIApiDeployments` array to the `CopilotSettings` interface in src/settings/model.ts.
- 📝 Ensure each deployment object in the array contains `deploymentName`, `instanceName`, `apiKey`, and `apiVersion` properties.
- 🔄 Update the `updateSetting` function in src/settings/model.ts to handle changes to the `azureOpenAIApiDeployments` array.
- 🗂️ Use the `azureOpenAIApiDeployments` array in the `ApiSettings` component in src/settings/components/ApiSettings.tsx to manage the deployments.
- 🔧 Ensure the `azureOpenAIApiDeployments` array is properly initialized in the `DEFAULT_SETTINGS` object in src/constants.ts.

What changes are needed in `src/LLMProviders/chainManager.ts` for Azure support?

To add Azure support in src/LLMProviders/chainManager.ts, the following changes are needed:

- 🛠️ Update the `createChainWithNewModel` method to handle Azure-specific configurations by checking if the provider is `azure openai` and setting the appropriate API key, instance name, and version.
- 📝 Ensure the `setChain` method properly initializes the chain with Azure-specific configurations when the provider is `azure openai`.
- 🔄 Modify the `getEffectivePrompt` method to handle Azure-specific prompts if needed.
- 🗂️ Update the `validateChatModel` method to include validation for Azure-specific configurations.
- 🔧 Ensure the `initializeQAChain` method properly initializes the QA chain with Azure-specific configurations when the provider is `azure openai`.

Dynamic deployment management

Add to task

- 🛠️ **Dynamic deployment management**: Implement a mechanism to dynamically add, edit, and remove Azure OpenAI deployments without requiring a full application restart. This can be achieved by updating the `azureOpenAIApiDeployments` array in the `CopilotSettings` interface in src/settings/model.ts.
- 📝 **Real-time updates**: Ensure that changes to the `azureOpenAIApiDeployments` array are immediately reflected in the application by using the `updateSetting` function in src/settings/model.ts. This will allow for real-time updates to the deployment settings.
- 🔄 **UI integration**: Integrate the deployment management functionality into the `ApiSettings` component in src/settings/components/ApiSettings.tsx. This will provide a user-friendly interface for managing deployments.
- 🗂️ **Deployment validation**: Implement validation checks to ensure that the deployment settings are correct before applying them. This can help prevent errors and ensure that the application is using valid configurations.
- 🔧 **Configuration updates**: Use the `updateModelConfig` function in src/aiParams.ts to apply the updated deployment settings to the model configuration. This will ensure that the correct settings are used when interacting with the Azure OpenAI API.

Centralized configuration

Add to task

- 🛠️ **Centralized configuration**: Define the `azureOpenAIApiDeployments` array in the `CopilotSettings` interface in src/settings/model.ts. This array should contain objects with properties like `deploymentName`, `instanceName`, `apiKey`, and `apiVersion`.
- 📝 **Update model configuration**: Modify the `updateModelConfig` function in src/aiParams.ts to handle Azure-specific configurations. This ensures that the correct deployment settings are applied when updating the model configuration.
- 🔄 **Update settings**: Use the `updateSetting` function in src/settings/model.ts to update the `azureOpenAIApiDeployments` array. This function will help manage changes to the deployment settings.
- 🗂️ **Manage deployments in UI**: Ensure the `azureOpenAIApiDeployments` array is used in the `ApiSettings` component in src/settings/components/ApiSettings.tsx to manage the deployments. This will allow users to add, edit, and remove deployments through the user interface.
- 🔧 **Retrieve model key**: Use the `getModelKey` function in src/aiParams.ts to retrieve the current model key and update the Azure-specific configurations accordingly.