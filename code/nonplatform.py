from __future__ import annotations

import re
import unicodedata

NL = chr(10)


def _fold_table():
    """A translate table for the Latin-1 range, built once.

    ``unicodedata.normalize`` per token costs about as much as everything else
    in this module put together, and the corpus is tens of millions of words.
    """
    t = {}
    for cp in range(0xC0, 0x180):
        ch = chr(cp)
        d = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in d if unicodedata.category(c) != "Mn")
        if base and base != ch:
            t[cp] = base
    return t


_FOLD = _fold_table()


def strip_accents(s):
    return str(s).translate(_FOLD)


def norm(s):
    return str(s).translate(_FOLD).lower()


def words(s):
    return len(re.findall(r"\w+", str(s)))


# --------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------

# Verbs a platform uses to propose an action.
ACTION = set("""
abrir acabar acompanhar adequar adotar adquirir agilizar ajudar alcancar
alfabetizar amplificar ampliar aperfeicoar aplicar apoiar apresentar articular
asfaltar assegurar assistir atender atrair atualizar aumentar ativar avaliar
beneficiar buscar cadastrar calcar capacitar cobrar colaborar colocar combater
conceder conhecer conquistar conservar consolidar construir continuar contratar
conversar coordenar criar cuidar cumprir dar defender descentralizar
desenvolver desburocratizar destinar diagnosticar dignificar diminuir
disponibilizar distribuir divulgar doar dobrar dotar duplicar efetivar elaborar
elevar eliminar emitir empregar encaminhar enfrentar engajar enxugar equipar
erradicar escoar estabelecer estender estimular estruturar estudar executar
exigir expandir facilitar fazer fechar firmar fiscalizar fomentar formar
fortalecer garantir gerar gerir humanizar implantar implementar incentivar
incluir incrementar informatizar iniciar inovar inserir instalar instituir
integrar isentar lancar legalizar levar liberar licitar lutar manter mapear
melhorar mobilizar modernizar monitorar montar negociar normatizar numerar
observar oferecer ofertar organizar orientar otimizar participar patrocinar
pavimentar perfurar permitir pesquisar planejar plantar pleitear possibilitar
potencializar praticar preservar prestigiar priorizar produzir profissionalizar
promover propor proporcionar proteger prover publicar qualificar readequar
reativar recuperar reduzir reeditar reequipar reestruturar reforcar reformar
regularizar remodelar renovar reorganizar reparar repassar requalificar
resgatar resolver respeitar restaurar retirar retomar revisar revitalizar
sanear ser simplificar sistematizar socializar subsidiar sustentar ter tornar
transferir transformar transportar trazer treinar unificar universalizar
urbanizar utilizar valorizar verificar viabilizar vincular visitar voltar zelar
zerar
acelerar administrar aprimorar auxiliar avancar conduzir determinar dirigir
fidelizar fundar governar identificar liderar ouvir pautar prospectar relocar
retomar trabalhar
""".split())

# Nouns that name an action; a proposal when a complement follows.
DEVERBAL = set("""
adequacao apoio aquisicao ampliacao amplificacao assistencia atendimento
aumento capacitacao conservacao construcao contratacao criacao distribuicao
doacao elaboracao execucao expansao fomento fortalecimento geracao implantacao
implementacao inclusao incentivo instalacao integracao investimento
investimentos isencao ligacao mapeamento manutencao melhoria melhorias
melhoramento modernizacao monitoramento oferta pavimentacao planejamento
plantio preservacao promocao protecao qualificacao realizacao reativacao
recapeamento recuperacao reducao reestruturacao reforma reformas renovacao
requalificacao
restauracao revitalizacao revisao saneamento sinalizacao subsidio transformacao
treinamento universalizacao urbanizacao valorizacao
""".split())

PREP = set("""de da do das dos na no nas nos em para a ao aos as com entre""".split())

# Bare policy-domain words: a line made only of these is a contents entry.
DOMAIN = set("""
abastecimento acessibilidade administracao administrativa agricultura
agronegocio agropecuaria ambiental ambiente apresentacao assistencia basico
basica ciencia cidadania comercio comunicacao conclusao consideracoes cultura
defesa desenvolvimento diretrizes economia economico educacao emprego
empreendedorismo energia esporte esportes estrutura familiar financas finais
gerais gestao habitacao idoso idosos indice industria infraestrutura infra
inovacao introducao juventude lazer meio mobilidade mulher mulheres municipal
municipalista obras participacao patrimonio planejamento politica politicas
prefeitura previdencia primeira programa proposta propostas publica publicas
publico qualidade renda rural saneamento saude seguranca servico servicos
social sociais sumario sustentabilidade tecnologia trabalho transito
transparencia transporte transportes turismo urbana urbanismo urbano vida zona
""".split())

STOPWORD = set("""e de da do das dos a o as os em para com no na nos nas ao aos
por sobre ou / , - : ; . ( ) i ii iii iv v vi vii viii ix x
""".split())

# Words that name a document's own structure or one of its administrative
# fields, never a promise.  A deverbal noun that follows one of these is its
# complement, not the head of a proposal: "Horário de atendimento ao público"
# and "Certidão de distribuição para fins gerais" name a thing, while
# "Construção da praça" and "Mais melhorias em ..." propose one.
STRUCTURE = set("""
anexo apendice apresentacao capitulo ciclo cronograma eixo eixos etapa fase
figura fluxo grafico indice item itens parte quadro secao sumario tabela tema
temas topico topicos
assunto ata atestado autor cartorio certidao certidoes classe comarca
comprovante declaracao denunciado edital emolumentos endereco ficha horario
horarios juizo local matricula numero objeto oficio portaria prazo processo
protocolo recibo requerente requerimento reu selo situacao telefone
""".split())

# A finite verb.  Accents are required where the tense carries one, and the
# -am/-em endings carry a stop list because Portuguese has a large family of
# "-agem" nouns.
FINITE_SUF = re.compile(
    r"\b\w{2,}(?:ar[áâ]|er[áâ]|ir[áâ]|ar[ãa]o|er[ãa]o|ir[ãa]o"
    r"|aria|eria|iria|ariam|eriam|iriam"
    r"|amos|emos|imos|omos|remos"
    r"|ou|aram|eram|iram|ava|avam|iam"
    r"|asse|esse|isse|assem|essem|issem)\b", re.I)
FINITE_WORD = re.compile(
    r"\b(?:é|s[ãa]o|ser[áa]|ser[ãa]o|foi|foram|est[áa]|est[ãa]o|h[áa]|tem|t[êe]m"
    r"|deve|devem|pode|podem|vamos|vai|v[ãa]o|iremos|queremos|precisa|precisamos"
    r"|garante|garantimos|faremos|temos|somos|seja|sejam|possa|possam|era|eram"
    r"|fica|ficam|existe|existem|visa|visam|busca|buscam|pretende|pretendemos"
    r"|prop[õo]e|propomos|apresenta|apresentamos|acredita|acreditamos|entendemos"
    r"|sabemos|estamos|consiste|consistem|representa|representam"
    r"|possui|possuem|contribui|constitui|oferece|realiza|atende|recebe"
    r"|permite|contempla|ocorre|torna|segue|inclui|cria|faz|diz|quer"
    # First person singular: a candidate writing "Pretendo", "Vou", "Declaro"
    # is writing a sentence, not listing a promise.
    r"|pretendo|vou|quero|tenho|sou|estou|declaro|firmo|prop[õo]nho|defendo"
    r"|acredito|entendo|considero|apresento|assumo|farei|irei|informo)\b", re.I)
AM_EM = re.compile(r"\b\w{3,}(?:am|em)\b", re.I)
AM_EM_STOP = set("""
abordagem alem alguem amperagem aprendizagem armazem bem contagem desvantagem
drenagem embalagem engrenagem estiagem ferragem folhagem forragem garagem
harem homem homenagem hospedagem ibidem idem imagem item jovem linguagem
malandragem margem mensagem modem montagem ninguem nuvem ordem origem paisagem
passagem porcentagem porem quilometragem reciclagem refem selvagem serragem
sondagem tambem totem triagem vantagem vertigem viagem virgem voltagem
""".split())

# First-person autobiography, and the appeal for a vote that closes one.
BIO = re.compile(
    r"\b(?:nasci|estudei|trabalhei|comecei|consegui|exerci|me formei|iniciei"
    r"|fui eleito|fui eleita|atuei|percorri|dediquei|aprendi|vivi)\b"
    r"|\bminha (?:trajet[óo]ria|hist[óo]ria|vida|carreira|caminhada)\b"
    r"|\bsou (?:filho|filha|natural|nascid)", re.I)
APPEAL = re.compile(
    r"voto de confian[çc]a|pe[çc]o (?:o seu|seu|um) voto|vote (?:em|no|na)\b"
    r"|conto com (?:o seu|seu|voc[êe])|obrigado pela sua aten[çc][ãa]o", re.I)


# --------------------------------------------------------------------------
# Text units
# --------------------------------------------------------------------------

HEADER = re.compile(r"^\s*#{1,6}\s")
BULLET = re.compile(r"^\s*(?:[-*•●▪■>]+|\d+[.)–-]|"
                    r"\d+\.\d+[.)]?|[a-z][.)]|\d+[º°])\s")
MARKUP = re.compile(r"[*_#>`|]+|<[^>]{1,40}>|\\+")
# "(3.1)" and "3.1" both number a proposal on the TSE form.
LEAD_NUM = re.compile(r"^\s*(?:\(?\d+(?:\.\d+)*\)?\s*[º°]?\s*[.)–-]?\s*)+")
SPLIT = re.compile(r"[;•●▪]")
BULLET_LINE = re.compile(r"^\s*[-*•●▪]|^\s*\d+[.)]\s")


def has_finite_verb(s):
    s = str(s)
    if FINITE_SUF.search(s) or FINITE_WORD.search(s):
        return True
    for m in AM_EM.finditer(s):
        if norm(m.group(0)) not in AM_EM_STOP:
            return True
    return False


def is_header(line):
    return bool(HEADER.match(str(line)))


def cells(line):
    """Non-empty cells of a markdown table row, or None if not a table row."""
    s = str(line).strip()
    if not s.startswith("|"):
        return None
    return [c.strip() for c in s.strip("|").split("|") if c.strip()]


def clean(line):
    """Strip bullets, numbering and markup so the first real token is first."""
    s = MARKUP.sub(" ", str(line))
    s = BULLET.sub(" ", s)
    s = LEAD_NUM.sub(" ", s)
    return s.strip(" \t.;:-–•")


def tokens(line):
    return re.findall(r"\w+", norm(line))


def segments(line):
    """A line split into the units a proposal can occupy."""
    return [p for p in SPLIT.split(str(line)) if p.strip()]


def is_number_only(line):
    t = tokens(line)
    return bool(t) and all(x.isdigit() for x in t)


def is_domain_label(line):
    """A line that names policy domains and nothing else: a contents entry."""
    t = [x for x in tokens(clean(line)) if x not in STOPWORD]
    if not t or len(t) > 6:
        return False
    return all(x in DOMAIN for x in t)


def is_list_marked(seg):
    """Does this segment carry a list marker of its own?

    A leading table pipe is stripped first: one platform lays its promises out
    as a one-column table, and those rows are a list, not data.
    """
    s = str(seg).lstrip().lstrip("|").lstrip()
    return bool(BULLET.match(s) or LEAD_NUM.match(s))


def opens_with_action(seg):
    t = tokens(clean(seg))
    if len(t) < 2 or t[0] in STRUCTURE:
        return False
    if t[0] in ACTION or t[1] in ACTION:
        return True
    for i in (0, 1, 2):
        if i >= len(t) or t[i] not in DEVERBAL:
            continue
        # A deverbal noun that complements a structural or administrative word
        # is not the head of a promise.
        if any(x in STRUCTURE for x in t[:i]):
            continue
        for j in range(i + 1, min(i + 5, len(t))):
            if t[j] not in PREP:
                continue
            # "Assistência Social e Desenvolvimento do Cidadão" names a policy
            # domain; "Assistência aos criadores de avicultura" proposes one.
            span = t[i + 1:j]
            if span and all(x in DOMAIN or x in STOPWORD for x in span):
                break
            return True
    return False


def is_proposal_segment(seg):
    """A proposal, as opposed to prose that happens to contain a verb.

    A proposal is either list-marked — a bullet, a number, a "2º -" inside a
    run-on paragraph — or a stretch of text that is not itself a sentence.
    Without the second condition, "Administrar os recursos públicos ... é a
    missão do governo municipal" reads as a proposal, and a placeholder that
    proposes nothing survives the contents-only arm.  There is no length limit:
    several platforms carry ninety promises on one unpunctuated line.
    """
    if not opens_with_action(seg):
        return False
    if is_list_marked(seg):
        return True
    return not has_finite_verb(seg)


def is_proposal(line):
    """Does any segment of this line propose something?

    A multi-column table row is handled separately by ``table_proposal``: a cell
    in a calendar is data about the document, but a cell in a platform laid out
    as a table is a promise about the city, and the two look identical here.
    """
    c = cells(line)
    if c is not None and len(c) >= 2:
        return False
    return any(is_proposal_segment(s) for s in segments(line))


def table_proposal(line):
    """Does a multi-column table row carry a proposal in one of its cells?"""
    c = cells(line)
    if c is None or len(c) < 2:
        return False
    return any(is_proposal_segment(s) for cell in c for s in segments(cell))


# --------------------------------------------------------------------------
# Marker sets, one per kind of non-platform
# --------------------------------------------------------------------------

# A: text ABOUT the campaign, addressed to the campaign team.  An earlier
# version keyed on "público alvo", "preenchimento" and "este espaço", which are
# ordinary platform vocabulary; it cut a 54,000-word platform.
CAMPAIGN = [
    r"caderno de campanha",
    r"question[áa]rio de planejamento",
    r"a[çc][õo]es de campanha devem ser",
    r"curr[íi]culo pol[íi]tico",
    r"p[úu]blico alvo chave da campanha",
    r"bandeiras est[ãa]o estruturadas",
    r"identifica[çc][ãa]o do principal (?:ponto forte|risco)",
    r"dom[íi]nio [ée] a capacidade que a candidatura",
    r"perfil de eleitorado adotado pelo",
    r"despender recursos.{0,40}eleitorado",
    r"discurso padronizado do candidato",
    r"como (?:quer|n[ãa]o quer) ser lembrad",
    r"a campanha (?:da|do|de) .{0,70} tem como foco",
    r"preencha|insira (?:aqui|seu|sua)|marque aqui|escale o seu time",
    r"modelo de plano de governo|use este modelo|utilize este modelo",
]

# B: a court, notarial or registry document.  Note what is NOT here: a citation
# of TSE Resolution 23.609 is not the form — real platforms cite it constantly.
JUDICIAL = [
    r"excelent[íi]ssim[oa]",
    r"ju[íi]z[oa]? (?:eleitoral|da \d+)",
    r"promotor(?:a)? eleitoral",
    r"minist[ée]rio p[úu]blico eleitoral",
    r"promotoria de justi[çc]a",
    r"manifesta-se o minist[ée]rio p[úu]blico",
    r"pede deferimento",
    r"v\.\s?exa\.",
    r"embargos de declara[çc][ãa]o",
    r"[óo]rg[ãa]o julgador",
    r"segredo de justi[çc]a",
    r"pedido de liminar",
    r"fiscal da lei",
    r"c[óo]digo eleitoral",
    r"nos autos|autos n[.º°]",
    r"inelegibilidade",
    r"procurador/terceiro vinculado",
    r"[úu]ltima distribui[çc][ãa]o",
    # Notarial: a power of attorney, or a declaration filed in place of a plan.
    r"outorgante", r"outorgad[oa]", r"ad judicia", r"substabelecer",
    r"oab/[a-z]{2}", r"firmo a presente", r"produza seus efeitos legais",
    r"para fins de registro de candidatura", r"nestes termos",
    r"relat[óo]rio de conhecimento", r"sisconta",
    r"vossa excel[êe]ncia", r"merit[íi]ssim",
    # The administrative paperwork a candidacy generates: clearance
    # certificates, filing receipts, consent forms, the candidate registry page,
    # a law firm's fee quotation.
    r"nada consta|n[ãa]o consta\b", r"certid[ãa]o (?:de|judicial|estadual|criminal)",
    r"recibo de peticionamento", r"certifico|certificamos",
    r"declara[çc][ãa]o de ci[êe]ncia", r"o referido [ée] verdade",
    r"para fins (?:exclusivamente )?eleitorais",
    r"distribui[çc][õo]es criminais", r"execu[çc][õo]es criminais",
    r"honor[áa]rios para a execu[çc][ãa]o", r"situa[çc][ãa]o candidatura",
]

# I: the party's convention minutes.
PARTY_ACT = [
    r"ata (?:da|de) conven[çc][ãa]o",
    r"edital de convoca[çc][ãa]o",
    r"conven[çc][ãa]o partid[áa]ria",
    r"nome urna:",
]

# J: the plan is not here; it will be filed later.  Every marker names the PLAN.
# "Será elaborado um projeto onde o morador ..." is a promise about the city and
# must not match.
DEFER = [
    r"(?:plano|proposta|programa|projeto|documento)[^.]{0,60}"
    r"(?:ser[áa] apresentad|ser[áa] constru[íi]d|ser[áa] elaborad|ser[áa] detalhad"
    r"|ser[áa] aperfei[çc]oad|ser[áa] complementad|est[áa] sendo constru[íi]d"
    r"|ainda ser[áa]|apresentar[áa])",
    r"em breve[^.]{0,80}(?:plano|proposta|apresenta)",
    r"plano de governo completo|plano completo",
    r"no decorrer da campanha",
    r"[áa]reas que ser[ãa]o contempladas",
    r"seguiremos os objetivos estrat[ée]gicos",
    r"n[ãa]o [ée] uma id[ée]ia final|n[ãa]o [ée] uma ideia final",
]

CAMPAIGN_RE = [re.compile(p, re.I) for p in CAMPAIGN]
JUDICIAL_RE = [re.compile(p, re.I) for p in JUDICIAL]
PARTY_RE = [re.compile(p, re.I) for p in PARTY_ACT]
DEFER_RE = [re.compile(p, re.I) for p in DEFER]

# An extract begins mid-document, at a numbered subsection.  A CNPJ has the same
# shape ("33.991.857/0001-70") and cut two platforms before it was excluded.
DEEPNUM = re.compile(r"^[\s#*]*(\d{1,2})\.(\d{1,2})(?!\d)")
CNPJ = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}")

# Below this length a document cannot hold a platform behind a docket, so arm B
# does not ask its judicial markers to reach deep into the text.
JUDICIAL_SHORT = 600

ARM_LABEL = {
    "A": "campaign workbook",
    "B": "court, notarial or registry filing",
    "C": "candidate letter or CV",
    "D": "no content",
    "E": "contents page and a letter",
    "F": "extract of a longer document",
    "G": "annex of tabulated data",
    "H": "shattered extraction",
    "I": "party convention minutes",
    "J": "plan to be filed later",
}


def n_distinct(text, patterns):
    return sum(1 for p in patterns if p.search(text))


def profile(text):
    """Every count the arms use, for one document."""
    text = str(text)
    lines = [l.strip() for l in text.split(NL)]
    lines = [l for l in lines if l]
    body = [l for l in lines if not is_header(l)]
    clause = [l for l in body if words(l) >= 8 and has_finite_verb(l)]
    label = [l for l in lines if is_domain_label(l)]
    number = [l for l in lines if is_number_only(l)]
    # Headings count as proposal sites: one platform puts "MELHORIAS PARA OS
    # BAIRROS" in a heading and its only proposal is there.
    prop = [l for l in lines if is_proposal(l)]
    table = [l for l in lines if (cells(l) or []) and len(cells(l)) >= 2]
    table_prop = [l for l in table if table_proposal(l)]
    item = [l for l in body
            if not is_domain_label(l) and not is_number_only(l)
            and cells(l) is None and 1 < words(l) <= 25]
    # An item that opens with a policy domain says something about the city:
    # "Saúde (Posto 24 horas)", "Educação (Cursos Técnicos)".
    domain_item = [l for l in item if (tokens(clean(l)) or [""])[0] in DOMAIN]
    return dict(lines=lines, body=body, clause=clause, label=label,
                number=number, prop=prop, table=table, item=item,
                table_prop=table_prop, domain_item=domain_item, nw=words(text))


def features(doc, text):
    """One flat row per document: everything ``arms_from_features`` needs.

    Kept separate from ``arms`` so a corpus scan can be cached and the
    thresholds re-decided without re-parsing tens of millions of words.
    """
    text = str(text)
    p = profile(text)
    lines = p["lines"]
    return dict(
        doc=doc, n_words=p["nw"], n_lines=len(lines),
        n_clause=len(p["clause"]), n_label=len(p["label"]),
        n_number=len(p["number"]), n_prop=len(p["prop"]),
        n_table=len(p["table"]), n_item=len(p["item"]),
        n_table_prop=len(p["table_prop"]),
        n_domain_item=len(p["domain_item"]),
        n_bullet=len([l for l in lines if BULLET_LINE.match(l)]),
        bio=len(BIO.findall(text)), appeal=bool(APPEAL.search(text)),
        campaign=n_distinct(text, CAMPAIGN_RE),
        judicial=n_distinct(text, JUDICIAL_RE),
        judicial_deep=max((m.start() for pat in JUDICIAL_RE
                           for m in pat.finditer(text)), default=-1) / max(1, len(text)),
        party_act=n_distinct(text, PARTY_RE),
        defer=n_distinct(text, DEFER_RE),
        first_line=(lines[0] if lines else "")[:200])


def arms_from_features(f):
    """The ten arms, from one feature row.  Returns the set of arm letters.

    Every arm is conjoined with ``noprop``: whatever else a document looks like,
    if it proposes something for the municipality it is a platform.
    """
    if not f["n_lines"]:
        return {"D"}
    noprop = not f["n_prop"] and not f["n_table_prop"]
    out = set()

    if f["campaign"] >= 2:
        out.add("A")

    # A court document is judicial THROUGHOUT.  A registration docket bound in
    # front of a platform carries its markers only at the top, and the platform
    # behind it is a platform — one such document runs the docket for a page and
    # then sets out a full platform in prose, which no proposal count catches —
    # so the last marker must fall past 40% of the text.  The depth test is
    # waived below JUDICIAL_SHORT words, because a document that short has no
    # room to hide a platform behind a docket.
    if (f["judicial"] >= 2 and noprop
            and (f["judicial_deep"] >= 0.4 or f["n_words"] < JUDICIAL_SHORT)):
        out.add("B")

    if (f["bio"] >= 3 and f["appeal"] and f["n_bullet"] <= 2
            and f["n_words"] < 2000 and noprop):
        out.add("C")

    # "No content" means the document does not so much as name a policy domain
    # in a list item.  Without that conjunct the arm removes a real platform
    # whose ten promises are written "Saúde (Posto 24 horas)".
    if not f["n_clause"] and noprop and not f["n_domain_item"]:
        out.add("D")

    # A contents page comes in two kinds, and the difference matters because a
    # short prose platform under four headings looks like neither.
    contents = (f["n_label"] >= 4 and noprop
                and f["n_item"] <= 5 and f["n_clause"] <= 6)
    if contents and (f["appeal"] or f["bio"] >= 1):
        out.add("E")
    if f["n_label"] >= 4 and noprop and f["defer"] >= 1:
        out.add("J")

    m = DEEPNUM.match(str(f["first_line"]))
    if (m and not CNPJ.search(str(f["first_line"]))
            and int(m.group(1)) <= 30 and int(m.group(2)) <= 30
            and (int(m.group(1)) > 1 or int(m.group(2)) > 1)):
        out.add("F")

    # An annex is a table of data.  A platform laid out as a table has a promise
    # in most of its rows; an annex has one in a third of them at most.
    if (f["n_table"] >= 10 and not f["n_prop"] and f["n_clause"] <= 6
            and f["n_table_prop"] < 0.35 * f["n_table"]):
        out.add("G")

    if (f["n_number"] >= 3 and f["n_number"] >= 0.15 * f["n_lines"]
            and not f["n_clause"]):
        out.add("H")

    if f["party_act"] >= 2 and noprop:
        out.add("I")

    return out


def arms(text, doc=""):
    """The arms a raw document fires."""
    return arms_from_features(features(doc, text))


def flag(text):
    """(does it fire, which arms) for one raw document."""
    a = sorted(arms(text))
    return bool(a), "+".join(a)
