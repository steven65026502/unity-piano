import socket
import json
class Dictionary:
    wordtypes = ['starttime', 'level', 'power', 'endtime']
    lenoftypes = {'starttime':32, 'level':88, 'power':128,'endtime':32}
    worddict = ['bar','endbar']
    for type in wordtypes:
        for index in range(lenoftypes[type]):
            worddict.append(f'{type},{index}')

    def onseteventstoword(pianoroll):
        part = 0
        output = []
        for event in pianoroll['onset_events']:
            for wordtype in wordtypes:
                usevalue = event[wordtypes.index(wordtype)]
                if (wordtype == 'starttime' or wordtype == 'endtime'):
                    endpart = 0
                    if wordtype == 'starttime':
                        while part <= int(usevalue / 32):
                            output.append('bar')
                            part += 1
                    else:
                        while part + endpart <= int(usevalue / 32):
                            output.append('endbar')
                            endpart += 1
                    usevalue = usevalue % 32
                elif wordtype == 'level':
                    usevalue -= 21
                output.append(f'{wordtype},{usevalue}')
        return output

    def wordtoonsetevents(words):
        part = -1
        endpart = 0
        onseteventsindex = -1
        output = {'onset_events':[], 'pedal_events':[]}
        for word in words:
            if word == 'bar':
                part += 1
            elif word == 'endbar':
                endpart += 1
            else:
                nowdata = word.split(',')
                if nowdata[0] == 'starttime':
                    output['onset_events'].append([0,0,0,0])
                    onseteventsindex += 1
                    nowdata[1] = int(nowdata[1]) + part * 32
                elif nowdata[0] == 'endtime':
                    nowdata[1] = int(nowdata[1]) + (part + endpart) * 32
                    endpart = 0
                elif nowdata[0] == 'level':
                    nowdata[1] = int(nowdata[1]) + 21
                output['onset_events'][onseteventsindex][wordtypes.index(nowdata[0])] = int(nowdata[1])
        return output

    def wordtoint(words):
        return [ worddict.index(a) for a in words ]

    def inttoword(wordints):
        return [ worddict[a] for a in wordints ]

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1",1234))
    s.listen()
    c, addr = s.accept()
    with c:
        print(addr,"connected.")

        while True:
            with open(input("json file:"), 'r') as f:
                musicdata = f.read().encode()
            #data = c.recv(1024)
            #if not data:
            #    break
            c.sendall(len(musicdata).to_bytes(4, 'big'))
            c.sendall(musicdata)
