import pandas as pd 
import os
from music21 import *

class Utility:
    def __init__(self) -> None:
        pass
    
    # Work for midi file with only one instrument (live input in particular)
    def midiToLstNote(self, midifilename="liveInput.mid"):
        stream_list=[]
        whole_piece=converter.parse(midifilename)
        for part in whole_piece.parts: # loads each channel/instrument into stream list
            partC= self.stream_to_C(part)            
            stream_list.append(partC)
        s_flat=stream_list[0].flat
        lst_note = []  
        for el in s_flat.notesAndRests: #for each note of the instrument
            n_temp= self.parseElement(el) # parse note, rest or chord
            n=(0,n_temp[0],n_temp[1]) # define node name
            lst_note.append(n)
        return lst_note
    
    def stream_to_C(self, part):
        k = part.flat.analyze('key')
        i = interval.Interval(k.tonic, pitch.Pitch('C'))
        part_transposed = part.transpose(i)
        return part_transposed
    
    def parseElement(self, el):
        p='other'
        if el.isNote:
            p=(str(el.pitch),str(el.quarterLength))
        if el.isRest:
            p=(el.name,str(el.quarterLength))
        if el.isChord:
            c=''
            for i in el.pitches:
                c+=str(i)+" "
            p=(c,str(el.quarterLength))
        return p
    
    def mergeDataToDF(self, df1, df2, list, songname):
        instrulst = []; str_instru = ""
        dic1 = self.dfToDic(df1); dic2 = self.dfToDic(df2); final_dic = {}
        # Create dic for instrument list
        for i in range(0, len(list)):
            name = self.cleanInstruName(list[i])
            if(name not in instrulst):
                instrulst.append(name)
                if(i < len(list)-1):
                    str_instru += name + ","
                else:
                    str_instru += name
        dic_tmp = {"Instruments": str_instru}
        
        final_dic["Name"] = songname
        final_dic["ID"] = abs(hash(songname))
        for key in dic1:
            if key not in final_dic:
                final_dic[key] = dic1[key]
        for key in dic2:
            if key not in final_dic:
                final_dic[key] = dic2[key]
        for key in dic_tmp:
            if key not in final_dic:
                final_dic[key] = dic_tmp[key]
        df = pd.DataFrame(final_dic, index=[0])
        return df
    
    def dfToDic(self, df):
        dic = df.to_dict('index')[0]
        return dic
    
    def cleanInstruName(self, name):
        name = name.replace(":","").replace("Instrument","").replace(",","").replace("TabIt MIDI - Track ","")
        name = ''.join([i for i in name if not i.isdigit()])
        name =' '.join(self.unique_list(name.split()))
        return name
    
    # https://stackoverflow.com/questions/7794208/how-can-i-remove-duplicate-words-in-a-string-with-python
    def unique_list(self,l):
        ulist = []
        [ulist.append(x) for x in l if x not in ulist]
        return ulist
    
    def checkDirExistAndCreate(self, outpath):
        isExist = os.path.exists(outpath)
        if not isExist:
            os.makedirs(outpath)
    
    def getCleanInstruName(self, rawname):
        cleanname = ["bass", "guitar", "piano", "percussion","sax", "organ"
                     "vibraphone","string", "voice","drum","flute","clarinet","harmonica",
                     "tuba", "brass", "sampler","synth", "oboe","banjo","cello","bassoon"
                     "horn", "trumpet", "keyboard"]
        for name in cleanname:
            if(name.replace(".","") in rawname.lower()):
                return name
                break
        return "Unknown"