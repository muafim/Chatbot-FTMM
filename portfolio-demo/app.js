/*
 * PORTFOLIO SHOWCASE ONLY
 * NO BACKEND CONNECTION
 * All responses and source records below are static presentation data.
 */

const conversation = document.querySelector("#conversation");
const composer = document.querySelector("#composer");
const questionInput = document.querySelector("#question-input");
const sourceList = document.querySelector("#source-list");
const sourceCount = document.querySelector("#source-count");
const sidebar = document.querySelector("#sidebar");
const menuButton = document.querySelector("#menu-button");
const sidebarScrim = document.querySelector("#sidebar-scrim");
const indexStatus = document.querySelector("#index-status");
const knowledgePopover = document.querySelector("#knowledge-popover");

const demoResponses = {
  dekan: {
    grounded: "Grounded in 1 FTMM source",
    answer: "Dekan FTMM adalah <strong>Prof. Dr. Dwi Setyawan, S.Si., M.Si., Apt.</strong> <button class=\"citation\" type=\"button\" data-source=\"dynamic-s1\" aria-label=\"Lihat sumber S1\">[S1]</button>",
    sources: [{ title: "Dwi Setyawan", meta: "Faculty · Dean", kind: "S1" }]
  },
  visi: {
    grounded: "Grounded in 1 FTMM source",
    answer: "Informasi visi dan misi tersedia pada sumber profil fakultas FTMM. Pada showcase ini, rincian teks tidak ditambahkan di luar cuplikan sumber yang telah diverifikasi. <button class=\"citation\" type=\"button\" data-source=\"dynamic-s1\" aria-label=\"Lihat sumber S1\">[S1]</button>",
    sources: [{ title: "Profil FTMM", meta: "Faculty · Visi dan Misi", kind: "S1" }]
  },
  nlp: {
    grounded: "Grounded in 3 FTMM sources",
    answer: "Sumber dosen menunjukkan keterkaitan <strong>Maryamah</strong> dan <strong>Muhammad Noor Fakhruzzaman</strong> dengan Natural Language Processing. <button class=\"citation\" type=\"button\" data-source=\"dynamic-s1\" aria-label=\"Lihat sumber S1\">[S1]</button> <button class=\"citation\" type=\"button\" data-source=\"dynamic-s2\" aria-label=\"Lihat sumber S2\">[S2]</button> Mata kuliah <strong>Natural Language Processing</strong> juga tersedia sebagai sumber terkait. <button class=\"citation\" type=\"button\" data-source=\"dynamic-s3\" aria-label=\"Lihat sumber S3\">[S3]</button>",
    sources: [
      { title: "Maryamah", meta: "Lecturer · Research Interest", kind: "S1" },
      { title: "Muhammad Noor Fakhruzzaman", meta: "Lecturer · Research Interest", kind: "S2" },
      { title: "Natural Language Processing", meta: "Course · Knowledge record", kind: "S3" }
    ]
  },
  skma: {
    grounded: "Grounded in 1 academic source",
    answer: "Informasi pengajuan SKMA tersedia dalam sumber informasi akademik FTMM. Showcase ini tidak menambahkan prosedur atau persyaratan yang tidak tercantum pada sumber. <button class=\"citation\" type=\"button\" data-source=\"dynamic-s1\" aria-label=\"Lihat sumber S1\">[S1]</button>",
    sources: [{ title: "Pengajuan SKMA", meta: "Academic · Procedure", kind: "S1" }]
  },
  machineLearning: {
    grounded: "Grounded in 1 course source",
    answer: "Sumber mata kuliah FTMM memuat <strong>Machine Learning</strong> sebagai knowledge record terkait. Detail kode, SKS, dan semester tidak ditampilkan pada respons showcase ini. <button class=\"citation\" type=\"button\" data-source=\"dynamic-s1\" aria-label=\"Lihat sumber S1\">[S1]</button>",
    sources: [{ title: "Machine Learning", meta: "Course · Knowledge record", kind: "S1" }]
  },
  fallback: {
    grounded: "Static showcase response",
    answer: "Demo interface ini menggunakan respons statis untuk preview portfolio. Coba pertanyaan tentang dekan, visi FTMM, NLP, SKMA, atau Machine Learning.",
    sources: []
  }
};

function escapeText(value) {
  const element = document.createElement("span");
  element.textContent = value;
  return element.innerHTML;
}

function responseFor(query) {
  const normalized = query.toLocaleLowerCase("id-ID");
  if (normalized.includes("dekan")) return demoResponses.dekan;
  if (normalized.includes("visi") || normalized.includes("misi")) return demoResponses.visi;
  if (normalized.includes("nlp") || normalized.includes("natural language")) return demoResponses.nlp;
  if (normalized.includes("skma")) return demoResponses.skma;
  if (normalized.includes("machine learning")) return demoResponses.machineLearning;
  return demoResponses.fallback;
}

function scrollConversation() {
  conversation.scrollTo({ top: conversation.scrollHeight, behavior: "smooth" });
}

function addUserMessage(query) {
  const article = document.createElement("article");
  article.className = "message user-message";
  article.innerHTML = `
    <div class="message-avatar user-avatar" aria-hidden="true">AM</div>
    <div class="message-body">
      <p class="message-author">You</p>
      <p>${escapeText(query)}</p>
    </div>`;
  conversation.append(article);
}

function addLoadingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant-message loading-message";
  article.id = "mock-loading";
  article.innerHTML = `
    <div class="message-avatar assistant-avatar" aria-hidden="true">FT</div>
    <div class="message-body">
      <span class="loading-label">Searching academic sources</span>
      <span class="loading-dots" aria-hidden="true"><i></i><i></i><i></i></span>
    </div>`;
  conversation.append(article);
  return article;
}

function renderSources(sources) {
  sourceCount.textContent = sources.length ? `${sources.length} referenced` : "No source selected";
  sourceList.innerHTML = "";

  sources.forEach((source, index) => {
    const card = document.createElement("button");
    card.className = "source-card";
    card.id = `dynamic-s${index + 1}`;
    card.type = "button";
    card.dataset.sourceCard = "";
    card.innerHTML = `
      <span class="source-index">${escapeText(source.kind)}</span>
      <span class="source-copy"><strong>${escapeText(source.title)}</strong><small>${escapeText(source.meta)}</small></span>
      <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>`;
    sourceList.append(card);
  });

  const demoCard = document.createElement("button");
  demoCard.className = "source-card demo-document";
  demoCard.type = "button";
  demoCard.dataset.sourceCard = "";
  demoCard.innerHTML = `
    <span class="source-index"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="M6 3h8l4 4v14H6zM14 3v5h5"/></svg></span>
    <span class="source-copy"><strong>Pedoman Akademik FTMM</strong><small>PDF · Halaman 12 · Demo source</small></span>
    <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>`;
  sourceList.append(demoCard);
}

function addAssistantMessage(response) {
  const article = document.createElement("article");
  article.className = "message assistant-message";
  article.innerHTML = `
    <div class="message-avatar assistant-avatar" aria-hidden="true">FT</div>
    <div class="message-body">
      <div class="assistant-meta">
        <p class="message-author">FTMM Academic Assistant</p>
        <span><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m7 12 3 3 7-7"/></svg>${escapeText(response.grounded)}</span>
      </div>
      <div class="answer-content">
        <p>${response.answer}</p>
        <p class="answer-note">Static portfolio response · No backend request was made.</p>
      </div>
    </div>`;
  conversation.append(article);
  renderSources(response.sources);
}

function highlightSource(sourceId) {
  document.querySelectorAll("[data-source-card]").forEach((card) => {
    card.classList.toggle("highlighted", card.id === sourceId);
  });
  document.querySelectorAll(".citation").forEach((citation) => {
    citation.classList.toggle("active", citation.dataset.source === sourceId);
  });
  const target = document.getElementById(sourceId);
  if (target) {
    target.focus({ preventScroll: true });
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = questionInput.value.trim();
  if (!query) return;

  addUserMessage(query);
  questionInput.value = "";
  questionInput.style.height = "auto";
  const loading = addLoadingMessage();
  scrollConversation();

  window.setTimeout(() => {
    loading.remove();
    addAssistantMessage(responseFor(query));
    scrollConversation();
  }, 520);
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 100)}px`;
});

document.addEventListener("click", (event) => {
  const citation = event.target.closest(".citation");
  if (citation) highlightSource(citation.dataset.source);

  const sourceCard = event.target.closest("[data-source-card]");
  if (sourceCard) highlightSource(sourceCard.id);
});

document.querySelectorAll("[data-demo-nav]").forEach((link) => {
  link.addEventListener("click", (event) => event.preventDefault());
});

document.querySelector("#new-chat").addEventListener("click", () => {
  questionInput.focus();
  document.body.classList.remove("sidebar-open");
});

menuButton.addEventListener("click", () => document.body.classList.add("sidebar-open"));
sidebarScrim.addEventListener("click", () => document.body.classList.remove("sidebar-open"));

indexStatus.addEventListener("click", () => {
  const willOpen = knowledgePopover.hidden;
  knowledgePopover.hidden = !willOpen;
  indexStatus.setAttribute("aria-expanded", String(willOpen));
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    document.body.classList.remove("sidebar-open");
    knowledgePopover.hidden = true;
    indexStatus.setAttribute("aria-expanded", "false");
  }
});

if (new URLSearchParams(window.location.search).get("screenshot") === "1") {
  document.body.classList.add("portfolio-shot");
}
