from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata


class ChatPDFBase:
    def __init__(self):
        self.vector_db_path = "chroma_db"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2024, chunk_overlap=500
        )
        self.prompt = ChatPromptTemplate.from_template(
            """You are an assistant that answers questions about the questions sent by the user about mining, algorithms...
            Here are some informations about the data extracted from the ontology, use this as truth
            Class: Algorithm
                Subclass: Exact
                Subclass: Heuristic
                    Subclass: GA
                    Subclass: HC
                    Subclass: RL
                    Subclass: TS
                    Subclass: Toposort
            ----------------------
            Class: Approach
                Subclass: Deterministic
                Subclass: Stochastic
            ----------------------
            Class: Author
            ----------------------
            Class: MathematicalModel
                Subclass: DES
                Subclass: LGP
                Subclass: MILP
                Subclass: -
            ----------------------
            Class: ObjectiveFunction
                Subclass: MonoObjective
                Subclass: MultiObjective
            ----------------------
            Class: ProblemType
                Subclass: ClusteringProblem
                    Subclass: BenchPhaseClustering
                    Subclass: LayerClustering
                    Subclass: MiningCutClustering
                Subclass: SchedulingProblem
            ----------------------
            Class: Publisher
            ----------------------
            Class: TimeHorizon
                Subclass: LTMP
                Subclass: STMP
            ----------------------
            Class: Title
            ----------------------
            Class: Year
            ----------------------
            === Object Properties ===
            All Title has_algorithm of Algorithmtype
            All Title has_approach of Approachtype
            All ClusteringProblem has_clustering_algorithm of Algorithmtype
            All Title has_clustering_problem of ClusteringProblemtype
            All Title has_objective of ObjectiveFunctiontype
            All Author has_paper of Titletype
            All Title has_publisher of Publishertype
            All SchedulingProblem has_scheduling_algorithm of Algorithmtype
            All SchedulingProblem has_scheduling_model of MathematicalModeltype
            All Title has_time_horizon of TimeHorizontype
            All Title has_year of Yeartype
            \n\n
        Here are the relevant excerpts from the papers (Title in the ontology): {context}\n Always respond in English.
        Question: {question}"""
        )
        # Inicialize o vector store Chroma com o diretório de persistência
        self.vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=FastEmbedEmbeddings(),
        )
        self.retriever = None
        self.chain = None

    def retrieve_relevant_chunks(self, query_text):

        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=FastEmbedEmbeddings(),
            )

        # Busca apenas os trechos mais relevantes (k=3, por exemplo)
        results = self.vector_store.similarity_search_with_relevance_scores(
            query_text, k=3
        )
        relevant_chunks = [doc.page_content for doc, score in results if score > 0.5]
        return "\n\n---\n\n".join(relevant_chunks)

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        print(pdf_file_path)
        if not docs:
            print(f"Nenhum documento carregado de {pdf_file_path}.")
            return  # Retorna se não houver documentos

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=FastEmbedEmbeddings(),
            )

        # Antes da transação
        estado_atual_embeddings = len(self.vector_store.get()["documents"])
        print("Total de embeddings registrados no ChromaDB:", estado_atual_embeddings)

        # Create unique IDs for each chunk
        unique_ids = [f"{pdf_file_path}_{i}" for i in range(len(chunks))]

        # Inserção de novos embeddings
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory=self.vector_db_path,
            ids=unique_ids,  # Pass the list of unique IDs
        )

        # Depois da transação
        estado_novo_embeddings = len(self.vector_store.get()["documents"])

        # Embeddings adicionados na transação
        adicionados = estado_novo_embeddings - estado_atual_embeddings

        print("Novos embeddings adicionados ao ChromaDB:", adicionados)

    def ask_com_db(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=FastEmbedEmbeddings(),
            )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        context = self.retrieve_relevant_chunks(query)
        prompt = self.prompt.format(context=context, question=query)
        print(prompt)
        return self.generate_response(prompt)

    def generate_response(self, prompt):
        """Método abstrato para gerar resposta. Deve ser implementado pelas subclasses."""
        raise NotImplementedError("Subclasses devem implementar este método.")

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
