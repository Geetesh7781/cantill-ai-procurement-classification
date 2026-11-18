
from __future__ import annotations
import argparse, json, re, sys, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    return pd.read_csv(path, low_memory=False)

def norm(x):
    if pd.isna(x): return ""
    s=str(x).lower().strip()
    s=re.sub(r'[^a-z0-9]+',' ',s)
    return re.sub(r'\s+',' ',s).strip()

def stable_key(prefix, value):
    value=norm(value)
    if not value: value="unknown"
    return prefix+"-"+hashlib.md5(value.encode()).hexdigest()[:10]

def taxonomy_rows(path):
    t=read_csv(path).fillna("")
    cols=[c for c in ["category_l1","category_l2","category_l3","category_l4"] if c in t.columns]
    return t[cols].astype(str).to_dict("records"), cols

def best_category(text, taxonomy, cols, train_text=None, train_labels=None):
    # Supervised nearest-neighbour when labels are available; tree-only semantic fallback otherwise.
    text=norm(text)
    if train_text is not None and len(train_text):
        vec=TfidfVectorizer(ngram_range=(1,2), analyzer="word", min_df=1, sublinear_tf=True)
        try:
            X=vec.fit_transform([norm(x) for x in train_text])
            q=vec.transform([text])
            sims=cosine_similarity(q,X).ravel()
            idx=int(np.argmax(sims))
            if sims[idx] >= 0.08:
                return train_labels[idx], float(min(.99,max(.20,sims[idx])))
        except Exception:
            pass
    # Build tree descriptions at run-time; no category names are hard-coded.
    docs=[]
    for row in taxonomy:
        parts=[row.get(c,"") for c in cols if row.get(c,"")]
        docs.append(norm(" ".join(parts)))
    if not docs: return {}, 0.0
    vec=TfidfVectorizer(ngram_range=(1,2), analyzer="word", sublinear_tf=True)
    try:
        X=vec.fit_transform(docs+[text])
        sims=cosine_similarity(X[-1],X[:-1]).ravel()
        idx=int(np.argmax(sims))
        score=float(sims[idx])
    except Exception:
        idx=0; score=0.0
    # Similarity score is deliberately conservative when descriptions are sparse.
    conf=min(.75,max(.05,score))
    return taxonomy[idx], conf


def batch_classify(texts, taxonomy, cols, train_text=None, train_labels=None):
    texts=[norm(x) for x in texts]
    if train_text:
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,max_features=120000)
        X=vec.fit_transform([norm(x) for x in train_text])
        Q=vec.transform(texts)
        sims=cosine_similarity(Q,X)
        ix=sims.argmax(axis=1); scores=sims.max(axis=1)
        return [(train_labels[int(j)],float(min(.99,max(.20,scores[i])))) if scores[i]>=.08 else best_category(texts[i],taxonomy,cols) for i,j in enumerate(ix)]
    docs=[" ".join(row.get(c,"") for c in cols if row.get(c,"")) for row in taxonomy]
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True)
    X=vec.fit_transform([norm(x) for x in docs+texts])
    sims=cosine_similarity(X[len(docs):],X[:len(docs)])
    return [(taxonomy[int(j)],float(min(.75,max(.05,float(score))))) for j,score in zip(sims.argmax(axis=1),sims.max(axis=1))]

def classify_with_seed(df, seed, taxonomy, cols, desc_col):
    if seed is None or seed.empty: return [best_category(x,taxonomy,cols) for x in df[desc_col].fillna("")]
    # Keep only included labelled examples with a category.
    seed=seed[(seed.get("include",1)==1)].copy()
    label_cols=[c for c in ["category_l1","category_l2","category_l3","category_l4"] if c in seed.columns]
    seed=seed.dropna(subset=[c for c in label_cols if c in seed.columns], how="all")
    texts=[]; labels=[]
    for _,r in seed.iterrows():
        texts.append(" ".join(str(r.get(c,"")) for c in [desc_col,"gl_account_description","material_number"] if c in r.index and pd.notna(r.get(c))))
        labels.append({c:str(r[c]) if pd.notna(r.get(c)) else "" for c in label_cols})
    # Also enrich with taxonomy strings, preserving exact tree labels.
    return [best_category(x,taxonomy,cols,texts,labels) for x in df[desc_col].fillna("")]

def spend_reason(status=None, desc="", gl="", doc="", amount=0):
    st=norm(status)
    if any(x in st for x in ["cancel","void"]): return "CANCELLED"
    if st and not any(x in st for x in ["approved","paid"]): return "NOT_APPROVED"
    blob=norm(" ".join([str(desc or ""),str(gl or ""),str(doc or "")]))
    nonproc=["intercompany","inter company","royalt","tax","vat","withholding","financing","interest","revenue","partner billing","dividend"]
    if any(x in blob for x in nonproc): return "NON_PROCUREMENT"
    return ""

def supplier_item_keys(df, supplier_col, item_col):
    # Deterministic normalized keys. Fuzzy supplier matching can be added without changing output schema.
    suppliers=df[supplier_col].fillna("").map(norm)
    items=df[item_col].fillna("").map(norm)
    return suppliers.map(lambda x: stable_key("SUP",x)), items.map(lambda x: stable_key("ITEM",x))

def output_frame(record_ids, include, reason, supplier, item, spend, cats, conf, cols):
    out=pd.DataFrame({
        "record_id":record_ids, "include":include, "exclusion_reason":reason,
        "supplier_key":supplier, "item_key":item, "spend":np.round(spend,2)
    })
    for i,c in enumerate(["l1","l2","l3","l4"]):
        src=cols[i] if i<len(cols) else None
        out[c]=[x.get(src,"") if src else "" for x in cats]
    out["confidence"]=np.round(conf,4)
    return out[["record_id","include","exclusion_reason","supplier_key","item_key","spend","l1","l2","l3","l4","confidence"]]

def run_a(data_dir, out_dir, zero_label=False):
    a=data_dir/"company_A"
    erp=read_csv(a/"erp_goods_receipts.csv")
    tool=read_csv(a/"invoice_tool_export.csv")
    taxonomy, cols=taxonomy_rows(a/"taxonomy_A.csv")

    # Labels are strictly optional in zero-label mode (including blind evaluation).
    labels_path = a/"labels_A.csv"
    if not zero_label and labels_path.exists():
        labels=read_csv(labels_path)
    else:
        labels=pd.DataFrame(columns=["record_id"] + cols + ["supplier_key", "item_key"])
    lab=labels.set_index("record_id") if "record_id" in labels.columns else pd.DataFrame().set_index(pd.Index([], name="record_id"))

    erp_ids=["ERP:"+str(x) for x in erp.line_id]
    erp_cats=[]; erp_sup=[]; erp_item=[]; erp_conf=[]
    for i, rid in enumerate(erp_ids):
        er=erp.iloc[i]
        if not zero_label and rid in lab.index:
            r=lab.loc[rid]
            # Duplicate record IDs should not silently produce a DataFrame here.
            if isinstance(r, pd.DataFrame):
                r=r.iloc[0]
            erp_cats.append({c:str(r[c]) if c in r.index and pd.notna(r[c]) else "" for c in cols})
            erp_sup.append(str(r.get("supplier_key", "")))
            erp_item.append(str(r.get("item_key", "")))
            erp_conf.append(.99)
        else:
            erp_cats.append({})
            supplier_value=er.get("supplier_name", "")
            item_value=" ".join(str(er.get(c, "")) for c in ["material_number", "material_short_text", "service_short_text"] if pd.notna(er.get(c, "")))
            erp_sup.append(stable_key("SUP", supplier_value))
            erp_item.append(stable_key("ITEM", item_value))
            erp_conf.append(0.0)

    if zero_label:
        erp_desc=[" ".join(str(er.get(c,"")) for c in ["material_short_text","service_short_text","material_number"] if pd.notna(er.get(c))) for _,er in erp.iterrows()]
        pred=batch_classify(erp_desc,taxonomy,cols)
        erp_cats=[x[0] for x in pred]; erp_conf=[x[1] for x in pred]
    # ERP is the primary source and its value is already net ZAR.
    erp_reason=[]; erp_include=[]; erp_spend=[]
    for _,r in erp.iterrows():
        reason=spend_reason(desc=r.get("material_short_text",""),gl=r.get("service_short_text",""),doc=r.get("spend_type",""),amount=r.get("value_zar",0))
        erp_reason.append(reason); erp_include.append(0 if reason else 1); erp_spend.append(float(r.get("value_zar",0) or 0) if not reason else 0)
    # Match invoice-tool records to ERP by normalized PO + supplier, then amount proximity.
    def po_norm(x):
        nums=re.findall(r'\d+',str(x)) if pd.notna(x) else []
        return "".join(nums[:1]) if nums else ""
    erp_po={}
    for i,r in erp.iterrows():
        p=po_norm(r.get("po_number"))
        if p: erp_po.setdefault(p,[]).append((i,r))
    tool_rows=[]; tool_reason=[]; tool_spend=[]; tool_sup=[]; tool_item=[]; tool_cats=[]; tool_conf=[]
    a_train_text=[]; a_train_labels=[]
    if not zero_label:
        for _, er in erp.iterrows():
            labid="ERP:"+str(er.get("line_id",""))
            if labid in lab.index:
                lr=lab.loc[labid]
                a_train_text.append(" ".join(str(er.get(c,"")) for c in ["material_short_text","service_short_text","material_number"] if pd.notna(er.get(c))))
                a_train_labels.append({c:str(lr[c]) if pd.notna(lr[c]) else "" for c in cols})
    for _,r in tool.iterrows():
        po=po_norm(r.get("po_reference"))
        candidates=erp_po.get(po,[]) if po else []
        matched=None
        if candidates:
            s=norm(r.get("supplier_name_as_captured",""))
            # best supplier + amount similarity
            best=-1
            net=float(pd.to_numeric(r.get("gross_amount"),errors="coerce") or 0)/(1+float(pd.to_numeric(r.get("vat_rate"),errors="coerce") if pd.notna(r.get("vat_rate")) else .15))
            for _,e in candidates:
                sim=fuzz.token_set_ratio(s,norm(e.get("supplier_name","")))/100
                amt=float(e.get("value_zar",0) or 0)
                close=1- min(1,abs(amt-net)/max(abs(amt),abs(net),1))
                score=.65*sim+.35*close
                if score>best: best=score; matched=e
        status=r.get("status","")
        reason=spend_reason(status=status,desc=r.get("line_text",""))
        if reason:
            inc=0; amount=0
        elif matched is not None and best>=.55:
            inc=0; reason="DUPLICATE"; amount=0
        else:
            inc=1; amount=float(pd.to_numeric(r.get("gross_amount"),errors="coerce") or 0)/(1+float(pd.to_numeric(r.get("vat_rate"),errors="coerce") if pd.notna(r.get("vat_rate")) else .15))
        tool_rows.append("TOOL:"+str(r.tool_record_id)); tool_reason.append(reason); tool_spend.append(amount)
        tool_cats.append({}); tool_conf.append(0.0)
        if matched is not None and best>=.55:
            # map supplier/item keys to matched ERP golden keys
            ix=matched.name
            tool_sup.append(erp_sup[ix]); tool_item.append(erp_item[ix])
        else:
            tool_sup.append(stable_key("SUP",r.get("supplier_name_as_captured","")))
            tool_item.append(stable_key("ITEM",r.get("line_text","")))
    # Duplicate invoice export rows: same invoice, PO, line text and gross amount; retain first.
    seen=set()
    for i,r in tool.iterrows():
        k=(norm(r.get("invoice_number")),po_norm(r.get("po_reference")),norm(r.get("line_text")),str(r.get("gross_amount")))
        if k in seen and tool_reason[i] not in ["CANCELLED","NOT_APPROVED","NON_PROCUREMENT"]:
            tool_reason[i]="DUPLICATE"; tool_spend[i]=0
        else: seen.add(k)
    tool_include=[0 if x else 1 for x in tool_reason]
    to_classify=[i for i,v in enumerate(tool_include) if v==1]
    if to_classify:
        predicted=batch_classify([tool.iloc[i].get("line_text","") for i in to_classify],taxonomy,cols,a_train_text,a_train_labels if a_train_text else None)
        for i,(cat,cf) in zip(to_classify,predicted):
            tool_cats[i]=cat; tool_conf[i]=cf
    # Combine outputs.
    erp_out=output_frame(erp_ids,erp_include,erp_reason,erp_sup,erp_item,erp_spend,erp_cats,erp_conf,cols)
    tool_out=output_frame(tool_rows,tool_include,tool_reason,tool_sup,tool_item,tool_spend,tool_cats,tool_conf,cols)
    out=pd.concat([erp_out,tool_out],ignore_index=True)
    out.to_csv(out_dir/"geetesh__A.csv",index=False)
    print("A rows",len(out),"included",int(out.include.sum()),"spend",round(out.spend.sum(),2))

def run_b(data_dir,out_dir):
    b=data_dir/"company_B"
    df=read_csv(b/"invoice_lines.csv")
    seed=read_csv(b/"seed_labels.csv")
    tax, cols=taxonomy_rows(b/"taxonomy_B.csv")
    fx=read_csv(b/"fx_rates_usd.csv")
    fxmap={(str(r.currency),str(r.month)):float(r.units_per_usd) for _,r in fx.iterrows()}
    dates=pd.to_datetime(df["posting_date"],errors="coerce")
    spends=[]; reasons=[]
    for i,r in df.iterrows():
        reason=spend_reason(desc=r.get("line_description",""),gl=r.get("gl_account_description",""),doc=r.get("document_type",""),amount=r.get("line_amount",0))
        # Credit/adjustment lines and known intercompany account patterns are assessed conservatively.
        blob=norm(str(r.get("gl_account",""))+" "+str(r.get("gl_account_description","")))
        if not reason and any(x in blob for x in ["ic05","ic06"]): reason="NON_PROCUREMENT"
        reasons.append(reason)
        amt=pd.to_numeric(r.get("line_amount"),errors="coerce")
        if pd.isna(amt): amt=0.0
        cur=str(r.get("currency","USD"))
        if cur=="USD": usd=float(amt)
        else:
            rate=pd.to_numeric(r.get("currency_rate"),errors="coerce")
            month=dates.iloc[i].strftime("%Y-%m") if pd.notna(dates.iloc[i]) else (pd.to_datetime(r.get("invoice_date"),errors="coerce").strftime("%Y-%m") if pd.notna(pd.to_datetime(r.get("invoice_date"),errors="coerce")) else "")
            if pd.isna(rate) or rate<=0: rate=fxmap.get((cur,month),np.nan)
            usd=float(amt)/float(rate) if pd.notna(rate) and rate>0 else 0.0
        spends.append(usd if not reason else 0.0)
    # Seed labels include line IDs and categories. Use labelled records as nearest-neighbour training.
    seed2=seed.copy()
    train=df.merge(seed2,on="line_id",how="inner",suffixes=("","_label"))
    train=train[train["include"]==1]
    desc_col="line_description"
    texts=[]; labels=[]
    for _,r in train.iterrows():
        texts.append(" ".join(str(r.get(c,"")) for c in ["line_description","material_number","gl_account_description"] if pd.notna(r.get(c))))
        labels.append({c:str(r[c]) if c in r and pd.notna(r[c]) else "" for c in ["category_l1","category_l2","category_l3"]})
    # Train on a balanced subset if necessary; 2,550 examples are manageable.
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True, max_features=120000)
    try:
        X=vec.fit_transform([norm(x) for x in texts])
        Q=vec.transform([" ".join(str(r.get(c,"")) for c in ["line_description","material_number","gl_account_description"] if pd.notna(r.get(c))) for _,r in df.iterrows()])
        sims=cosine_similarity(Q,X)
        idx=sims.argmax(axis=1); score=sims.max(axis=1)
        cats=[labels[j] for j in idx]; conf=[float(min(.99,max(.15,score[i]))) for i in range(len(df))]
    except Exception:
        cats=[]; conf=[]
        for _,r in df.iterrows():
            c,s=best_category(" ".join(str(r.get(c,"")) for c in ["line_description","material_number","gl_account_description"]),tax,cols)
            cats.append(c); conf.append(s)
    # Supplier and item keys based on vendor ID and material/description.
    sup=[stable_key("SUP",x) for x in df["vendor_id"].fillna("")]
    item=[stable_key("ITEM", (str(r.get("material_number",""))+" "+str(r.get("line_description",""))).strip()) for _,r in df.iterrows()]
    include=[0 if x else 1 for x in reasons]
    spend=[spends[i] if include[i] else 0 for i in range(len(df))]
    out=output_frame(df.line_id.astype(str).tolist(),include,reasons,sup,item,spend,cats,conf,cols)
    out.to_csv(out_dir/"geetesh__B.csv",index=False)
    print("B rows",len(out),"included",int(out.include.sum()),"spend USD",round(out.spend.sum(),2))



def run_c(data_dir,out_dir,config_path=None):
    cfg_path=Path(config_path) if config_path else ROOT/"config/company_C.json"
    cfg=json.loads(cfg_path.read_text(encoding="utf-8"))
    df=read_csv(data_dir/cfg["input_file"])
    taxdf=read_csv(data_dir/cfg["taxonomy_file"]).fillna("")
    cat_cols=cfg.get("taxonomy_columns", [c for c in taxdf.columns if c.lower().startswith("category")])
    taxonomy=taxdf[cat_cols].astype(str).to_dict("records")
    f=cfg["fields"]
    desc=df.get(f.get("description",""),pd.Series([""]*len(df))).fillna("")
    supplier=df.get(f.get("supplier",""),pd.Series([""]*len(df))).fillna("")
    item=df.get(f.get("item",""),pd.Series([""]*len(df))).fillna("")
    status=df.get(f.get("status",""),pd.Series([""]*len(df))).fillna("")
    amount=pd.to_numeric(df.get(f.get("amount",""),pd.Series([0]*len(df))),errors="coerce").fillna(0)
    date=pd.to_datetime(df.get(f.get("date",""),pd.Series([None]*len(df))),errors="coerce")
    currency_col=f.get("currency")
    currency=df[currency_col].fillna(cfg.get("reporting_currency","USD")) if currency_col in df else pd.Series([cfg.get("reporting_currency","USD")]*len(df))
    rate_col=f.get("currency_rate")
    rates=cfg.get("fx_rates",{})
    spend=[]; reasons=[]
    for i,r in df.iterrows():
        reason=spend_reason(status=status.iloc[i],desc=desc.iloc[i],gl=r.get(f.get("account_description",""),""),doc=r.get(f.get("document_type",""),""))
        reasons.append(reason)
        val=float(amount.iloc[i])
        cur=str(currency.iloc[i]); report=cfg.get("reporting_currency","USD")
        if cur != report:
            rate=pd.to_numeric(r.get(rate_col),errors="coerce") if rate_col and pd.notna(r.get(rate_col)) else np.nan
            if pd.isna(rate) or rate<=0:
                month=date.iloc[i].strftime("%Y-%m") if pd.notna(date.iloc[i]) else ""
                rate=rates.get(cur,{}).get(month,rates.get(cur,{}).get("default",np.nan))
            val=val/float(rate) if pd.notna(rate) and float(rate)>0 else 0.0
        spend.append(val if not reason else 0.0)
    cats=[]; conf=[]
    for x in desc:
        c,s=best_category(x,taxonomy,cat_cols)
        cats.append(c); conf.append(s)
    sup=[stable_key("SUP",x) for x in supplier]
    itm=[stable_key("ITEM",x) for x in item]
    include=[0 if x else 1 for x in reasons]
    ids=df[f.get("record_id","line_id")].astype(str).tolist()
    out=output_frame(ids,include,reasons,sup,itm,spend,cats,conf,cat_cols)
    out.to_csv(out_dir/"geetesh__C.csv",index=False)
    print("C rows",len(out),"included",int(out.include.sum()),"spend",round(out.spend.sum(),2))

def main():
    ap=argparse.ArgumentParser(description="Cantill.ai configurable procurement spend pipeline")
    ap.add_argument("--company",choices=["A","B","C","all"],default="all")
    ap.add_argument("--data-dir",default=str(ROOT/"data"))
    ap.add_argument("--output-dir",default=str(ROOT/"outputs"))
    ap.add_argument("--zero-label",action="store_true",help="Run A classification without labels_A.csv")
    args=ap.parse_args()
    data=Path(args.data_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if args.company in ("A","all"): run_a(data,out,args.zero_label)
    if args.company in ("B","all"): run_b(data,out)
    if args.company=="C": run_c(data,out)

if __name__=="__main__": main()
