const express = require("express");
const response = express.Router().use(express.json(), express.urlencoded({ extended: false }));
require("cheerio");
const { CheerioWebBaseLoader } = require("langchain/document_loaders/web/cheerio");
const {RecursiveCharacterTextSplitter} = require("langchain/text_splitter");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { OpenAIEmbeddings, ChatOpenAI } = require("@langchain/openai");
const { pull } = require("langchain/hub");
const { ChatPromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");
const { createStuffDocumentsChain } = require("langchain/chains/combine_documents");

response.post("/", async (req, res)=> {

    const chat = req.body.content;

    //console.log("chat", chat)

    const loadDoc = new CheerioWebBaseLoader(
        "https://docs.google.com/document/d/1Jqq8bTQFQhGvnnhq-FmaIA-CpcWUpsdlUIJzW0S1smI/edit?usp=sharing"
    )

    const _3mttDoc = await loadDoc.load()

    const textSplitter = new RecursiveCharacterTextSplitter({chunkSize: 1000, chunkOverlap: 200});
    const splits = await textSplitter.splitDocuments(_3mttDoc);
    const vectorStore = await MemoryVectorStore.fromDocuments(splits, new OpenAIEmbeddings())

    //retrieve info from snippets of 3mtt doc and generate a response

    const retriever = vectorStore.asRetriever();
    const prompt = await pull("rlm/rag-prompt");
    const llm = new ChatOpenAI({modelName: "gpt-3.5-turbo", temperature: 0})
    
    const ragChain = await createStuffDocumentsChain({
        llm,
        prompt,
        outputParser: new StringOutputParser()
    })

    const retrievedDocs = await retriever.getRelevantDocuments(chat)

    const genOutput = await ragChain.invoke({
        question: chat,
        context: retrievedDocs,
    })

    //console.log(genOutput)  

    res.send(genOutput)

})

module.exports = response