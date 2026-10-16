import{_ as o}from"./slidev/CodeBlockWrapper.vue_vue_type_script_setup_true_lang-Y6JzWGb8.js";import{o as r,b as c,w as n,g as s,d,m as u,ad as e,v as m,x as h,T as l}from"./modules/vue-BOv15R7L.js";import{I as _}from"./slidev/default-CcpGrEgI.js";import{u as f,f as g}from"./slidev/context-CcvBypts.js";import"./modules/unplugin-icons-B5zimWMt.js";import"./index-B7XA2VA6.js";import"./modules/shiki-RcwD2jZ3.js";const y={__name:"gqa-decode-slides.md__slidev_5",setup(P){const{$clicksContext:t,$frontmatter:p}=f();return t.setup(),(k,a)=>{const i=o;return r(),c(_,m(h(l(g)(l(p),4))),{default:n(()=>[a[1]||(a[1]=s("h2",null,"Three-Phase Plan",-1)),d(i,u({},{title:"",ranges:[]}),{default:n(()=>[...a[0]||(a[0]=[s("pre",{class:"shiki shiki-themes vitesse-dark vitesse-light slidev-code",style:{"--shiki-dark":"#dbd7caee","--shiki-light":"#393a34","--shiki-dark-bg":"#121212","--shiki-light-bg":"#ffffff"}},[s("code",{class:"language-text"},[s("span",{class:"line"},[s("span",null,"Phase 1: MHA n_heads > 1")]),e(`
`),s("span",{class:"line"},[s("span",null,"  Fix score X-reduce to be head-scoped (pes_p_head PEs)")]),e(`
`),s("span",{class:"line"},[s("span",null,"  gqa_group_size = 1 degenerate case of Phase 2")]),e(`
`),s("span",{class:"line"},[s("span")]),e(`
`),s("span",{class:"line"},[s("span",null,"          ↓")]),e(`
`),s("span",{class:"line"},[s("span")]),e(`
`),s("span",{class:"line"},[s("span",null,"Phase 2: GQA + Option C")]),e(`
`),s("span",{class:"line"},[s("span",null,"  Compact K^T/V (kv_dim_p_pe rows per PE, not replicated)")]),e(`
`),s("span",{class:"line"},[s("span",null,"  W_Q/W_O offline column permutation → interleaved Q layout")]),e(`
`),s("span",{class:"line"},[s("span",null,"  KV-head-scoped X-reduce (pes_p_kv_head PEs)")]),e(`
`),s("span",{class:"line"},[s("span")]),e(`
`),s("span",{class:"line"},[s("span",null,"          ↓")]),e(`
`),s("span",{class:"line"},[s("span")]),e(`
`),s("span",{class:"line"},[s("span",null,"Phase 3: Incremental KV Cache")]),e(`
`),s("span",{class:"line"},[s("span",null,"  On-chip cache management (process_kv, iter_num)")]),e(`
`),s("span",{class:"line"},[s("span",null,"  Inherits Phase 2 compute layout unchanged")]),e(`
`),s("span",{class:"line"},[s("span",null,"  Adds variable-length GEMV variants")])])],-1)])]),_:1},16),a[2]||(a[2]=s("p",null,[s("strong",null,"Rationale:"),e(" Phase 2 degenerates to Phase 1 when "),s("code",null,"gqa_group_size = 1"),e(". Phase 1 tests validate Phase 2 code for free.")],-1))]),_:1},16)}}};export{y as default};
