from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from bs4 import BeautifulSoup
import re
from langchain_community.docstore.document import Document
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ProcessPoolExecutor
import pickle
import gc
import os
import concurrent.futures
import gc

def get_data(pat):
    loader = PDFMinerPDFasHTMLLoader(pat)
    data = loader.load()[0]
    return data

def get_content(data):
    soup = BeautifulSoup(data.page_content, 'html.parser')
    content = soup.find_all('span')
    return content


def get_snippets(content):
    import re
    cur_hSimNum = None
    cur_fsm = None
    cur_fs = None
    cur_text = ''
    snippets = []   # first collect all snippets that have the same font size
    for c in content:
        st = c.get('style')
        if not st:
            continue
        fsm = re.findall('font-family:(.*?);',st)
        fs = re.findall('font-size:(\d+)px',st) 
        text_slice= c.text[:5]
        hSimNum = re.findall(r'\w+\.\w*|§', text_slice)
        if not fsm or not fs:
            continue
        fsm = str(fsm[0]).strip()
        fs = int(fs[0])
        if hSimNum:
            hSimNum = 't'
        else:
            hSimNum = 'f'   
        if not cur_fsm: 
            cur_fsm = fsm
        if not cur_fs:
            cur_fs = fs  
        if not cur_hSimNum:
            cur_hSimNum = hSimNum        
        if fsm == cur_fsm and fs == cur_fs and hSimNum == cur_hSimNum: 
            cur_text += c.text   
        else:
            snippets.append((cur_text,cur_fsm,cur_fs,cur_hSimNum))
            cur_fsm = fsm
            cur_fs = fs
            cur_hSimNum = hSimNum
            cur_text = c.text
    snippets.append((cur_text,cur_fsm,cur_fs,cur_hSimNum))
    return snippets    

def get_semantic_snippets(data,snippets):
    cur_idx = -1
    semantic_snippets = []
    # Assumption: headings have higher font size than their respective content
    for s in snippets:
        # if current snippet's font size > previous section's heading => it is a new heading
        if not semantic_snippets or s[2] > semantic_snippets[cur_idx].metadata['heading_fontS']:   
            metadata={'heading':s[0],'content_fontM':'', 'heading_fontM':s[1], 'content_fontS': 0, 'heading_fontS': s[2], 'content_has_h_elem':'', 'heading_has_h_elem':s[3]}
            metadata.update(data.metadata)
            semantic_snippets.append(Document(page_content='',metadata=metadata))
            cur_idx += 1
            continue
        
        if s[1] != semantic_snippets[cur_idx].metadata['content_fontM']:
            if semantic_snippets[cur_idx].page_content != '':
                if 'Bold' in s[1] or 'Italic' in s[1] or 'BoldItalic' in s[1] or 'BoldOblique' in s[1] or 'BoldOblique' in s[1]:
                    metadata={'heading':s[0],'content_fontM':'', 'heading_fontM':s[1], 'content_fontS': 0, 'heading_fontS': s[2], 'content_has_h_elem':'', 'heading_has_h_elem':s[3]}
                    metadata.update(data.metadata)
                    semantic_snippets.append(Document(page_content='',metadata=metadata))
                    cur_idx += 1
                    continue
                    
        
        if s[3] == 't':
            if semantic_snippets[cur_idx].page_content != '':
                metadata={'heading':s[0],'content_fontM':'', 'heading_fontM':s[1], 'content_fontS': 0, 'heading_fontS': s[2], 'content_has_h_elem':'', 'heading_has_h_elem':s[3]}
                metadata.update(data.metadata)
                semantic_snippets.append(Document(page_content='',metadata=metadata))
                cur_idx += 1
                continue
        # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
        # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)

            
        if not semantic_snippets[cur_idx].metadata['content_fontS'] or not semantic_snippets[cur_idx].metadata['content_fontM'] or not semantic_snippets[cur_idx].metadata['content_has_h_elem'] or not 'Bold' in s[1] or not 'Italic' in s[1] or not 'BoldItalic' in s[1] or not 'BoldOblique' in s[1] or not  'BoldOblique' in s[1] or s[2] <= semantic_snippets[cur_idx].metadata['content_fontS']:
            semantic_snippets[cur_idx].page_content += s[0]
            semantic_snippets[cur_idx].metadata['content_fontM'] = s[1] if not semantic_snippets[cur_idx].metadata['content_fontM'] else semantic_snippets[cur_idx].metadata['content_fontM']
            semantic_snippets[cur_idx].metadata['content_fontS'] = max(s[2], semantic_snippets[cur_idx].metadata['content_fontS'])
            semantic_snippets[cur_idx].metadata['content_has_h_elem'] = s[3]
            continue

        # if current snippet's font size > previous section's content but less than previous section's heading than also make a new
        # section (e.g. title of a PDF will have the highest font size but we don't want it to subsume all sections)
        

        
        if s[2] > semantic_snippets[cur_idx].metadata['content_fontS'] and s[2] < semantic_snippets[cur_idx].metadata['heading_fontS']:
            metadata={'heading':s[0],'content_fontM':'', 'heading_fontM':s[1], 'content_fontS': 0, 'heading_fontS': s[2], 'content_has_h_elem':'','heading_has_h_elem':'t'}
            metadata.update(data.metadata)
            semantic_snippets.append(Document(page_content='',metadata=metadata))
            cur_idx += 1
            continue
        
        
        
        metadata={'heading':s[0], 'content_fontM':'', 'heading_fontM':s[1],'content_fontS': 0, 'heading_fontS': s[2],'content_has_h_elem':'','heading_has_h_elem':'t'}
        metadata.update(data.metadata)
        semantic_snippets.append(Document(page_content='',metadata=metadata))
        cur_idx += 1 
    return semantic_snippets
        
def process_file(file_path, file_name):
    gc.collect()
    data = get_data(file_path)
    semantic_snippets = get_semantic_snippets(data,get_snippets(get_content(data)))
    save_path = f'D:/school/ALL_PDFs/sniptemp/{file_name}.pkl'
    with open(save_path, 'wb') as outp:
        pickle.dump(semantic_snippets, outp, pickle.HIGHEST_PROTOCOL)
    gc.collect()
    
def run(file_paths, file_names):
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, file_paths, file_names)       