import { access, readFile } from 'node:fs/promises'
import { fileURLToPath, URL } from 'node:url'

import ansis from 'ansis'
import dotenv from 'dotenv'
import { glob } from 'tinyglobby'

import { MarkdownTextSplitter } from 'langchain/text_splitter'

import { HNSWLib } from '@langchain/community/vectorstores/hnswlib'
import { ChatAnthropic } from '@langchain/anthropic'
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts'
import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages'
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents'
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever'
import { createRetrievalChain } from 'langchain/chains/retrieval'

import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/huggingface_transformers'
import { intro, outro, text, isCancel, spinner, log } from '@clack/prompts'

const DOCS_DIR = fileURLToPath(new URL('../../docs', import.meta.url))
const VECTOR_STORE_DIR = fileURLToPath(new URL('../../.vector_store', import.meta.url))

const SYSTEM__PROMPT = `
You are a helpful Netlify CLI assistant that answers questions about this tool.

FORMAT YOUR RESPONSES FOR A COLOR TERMINAL:
1. Use ANSI color escape sequences to enhance your replies:
   - Use cyan for headings
   - Use yellow for code or commands
   - Use green for success or examples
   - Use red for warnings or errors
   - Use bold for emphasis
   - Always end colored sections with escape sequences to reset formatting

2. Structure your responses for terminal reading:
   - Keep paragraphs short and readable on narrow screens
   - Use clear section headers in cyan
   - When showing commands, format them in yellow
   - When showing example output, format in green
   - Use plain text without Markdown formatting

Your answers should be concise, accurate, and formatted for easy terminal reading.

If you don't know the answer, just say that you don't know.`

// Load env variables
dotenv.config()

const exists = async (file: string): Promise<boolean> => {
  try {
    await access(file)
    return true
  } catch {
    return false
  }
}

async function setupVectorStore(): Promise<HNSWLib> {
  const spin = spinner()
  spin.start('Loading documentation')

  // Read markdown files from docs directory
  const files = await glob(`${DOCS_DIR}/**/*.md`)

  let allDocs = ''
  for (const file of files) {
    const content = await readFile(file, 'utf8')
    allDocs += content + '\n\n'
  }

  // Split text into chunks
  const splitter = new MarkdownTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  })
  const docs = await splitter.createDocuments([allDocs])

  console.warn = () => {}

  // Use Hugging Face embeddings (local, no API required)
  const embeddings = new HuggingFaceTransformersEmbeddings({
    model: 'Xenova/all-MiniLM-L6-v2',
  })

  // Store files in a directory
  let vectorStore: HNSWLib

  if (await exists(VECTOR_STORE_DIR)) {
    // Load existing store
    vectorStore = await HNSWLib.load(VECTOR_STORE_DIR, embeddings)
  } else {
    // Create new store
    vectorStore = await HNSWLib.fromDocuments(docs, embeddings)
    await vectorStore.save(VECTOR_STORE_DIR)
  }

  spin.stop('Documentation loaded and indexed')
  return vectorStore
}

export async function startAIHelp(): Promise<void> {
  intro('Netlify CLI Interactive Help')

  let vectorStore: HNSWLib
  try {
    vectorStore = await setupVectorStore()
  } catch (error) {
    console.debug(error)
    log.error((error as Error).message)
    return
  }

  const llm = new ChatAnthropic({
    modelName: 'claude-3-7-sonnet-20250219',
    temperature: 0.2,
  })

  // Contextualize question
  const contextualizeQSystemPrompt = `
  Given a chat history and the latest user question
  which might reference context in the chat history,
  formulate a standalone question which can be understood
  without the chat history. Do NOT answer the question, just
  reformulate it if needed and otherwise return it as is.`
  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ['system', contextualizeQSystemPrompt],
    new MessagesPlaceholder('chat_history'),
    ['human', '{input}'],
  ])
  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm,
    retriever: vectorStore.asRetriever(),
    rephrasePrompt: contextualizeQPrompt,
  })

  // Answer question
  const qaSystemPrompt = `${SYSTEM__PROMPT}
  \n\n
  {context}`
  const qaPrompt = ChatPromptTemplate.fromMessages([
    ['system', qaSystemPrompt],
    new MessagesPlaceholder('chat_history'),
    ['human', '{input}'],
  ])

  // Below we use createStuffDocuments_chain to feed all retrieved context
  // into the LLM. Note that we can also use StuffDocumentsChain and other
  // instances of BaseCombineDocumentsChain.
  const questionAnswerChain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt,
  })

  const ragChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: questionAnswerChain,
  })

  const chatHistory: BaseMessage[] = []

  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  while (true) {
    const query = await text({
      message: 'How can I help you ship?\n',
      placeholder: 'Ask a question about using the Netlify CLI or Netlify in general...',
    })

    if (isCancel(query)) {
      break
    }

    if (typeof query === 'string' && query.toLowerCase() === 'exit') {
      break
    }

    const spin = spinner()
    spin.start('Thinking...')

    try {
      const response = await ragChain.invoke({
        chat_history: chatHistory,
        input: query,
      })
      const answer = response.answer
        .replaceAll('\\x1b', '\x1b')
        .replaceAll('\\u001b', '\u001b')
        .replaceAll('\\033', '\x1b')

      spin.stop()
      log.step(answer)

      chatHistory.push(new HumanMessage(query), new AIMessage(answer))
    } catch (error) {
      spin.stop('Error')
      log.error(`Error getting response: ${(error as Error).message}`)
    }
  }

  outro(ansis.blue('Thanks for using the Netlify CLI Interactive Help - now ship it!'))
}
