using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Linq;
using UnityEngine;

public class Manager : MonoBehaviour
{
    public const int INT_LEN = 4;

    private static Manager _instance = null;
    public static Manager Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = new GameObject("Manager").AddComponent<Manager>();
                DontDestroyOnLoad(_instance.gameObject);
            }
            return _instance;
        }
    }

    private List<Action> doinupdate;
    public List<Dictionary<string, object>> timeAndNotes;
    public List<List<Dictionary<string, object>>> ServerTimeAndNotes;
    private Socket socket;
    public float time { get; private set; } = -1;

    private void Awake()
    {
        time = -1;
        doinupdate = new List<Action>();
        timeAndNotes = new List<Dictionary<string, object>>();
        ServerTimeAndNotes = new List<List<Dictionary<string, object>>>();
        socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        socket.BeginConnect(new IPEndPoint(IPAddress.Loopback, 1234), (ar) =>
        {
            socket.EndConnect(ar);
            byte[] len = new byte[4];
            socket.BeginReceive(len, 0, INT_LEN, SocketFlags.None, Response, len);
        }, null);
    }

    public void TimeStart()
    {
        if(time < 0) time = 0;
    }

    public void TimeStop()
    {
        time = -1;
    }

    public byte[] HToNAndNToH(byte[] host)
    {
        byte[] bytes = new byte[host.Length];
        host.CopyTo(bytes, 0);

        if (BitConverter.IsLittleEndian)
            Array.Reverse(bytes);

        return bytes;
    }

    private void Response(System.IAsyncResult asyncResult)
    {
        byte[] lendata = (byte[])asyncResult.AsyncState;
        socket.EndReceive(asyncResult);
        if (lendata.Length < 1)
        {
            socket.Close();
            return;
        }
        int len = System.BitConverter.ToInt32(HToNAndNToH(lendata));
        byte[] data = new byte[len];
        if (socket.Receive(data, 0, len, SocketFlags.None) < 1)
        {
            socket.Close();
            return;
        }
        lock (ServerTimeAndNotes)
        {
            if (ServerTimeAndNotes.Count <= 0)
                TimeStop();
            ServerTimeAndNotes.Add(JsonConvert.DeserializeObject<List<Dictionary<string, object>>>(System.Text.Encoding.UTF8.GetString(data)));
        }
        socket.BeginReceive(lendata, 0, INT_LEN, SocketFlags.None, Response, lendata);
    }

    private void Update()
    {
        lock (doinupdate)
        {
            while (doinupdate.Count != 0)
            {
                Action action = doinupdate[0];
                doinupdate.RemoveAt(0);
                action();
            }
        }

        lock (ServerTimeAndNotes)
        {
            if (ServerTimeAndNotes.Count > 0)
            {
                TimeStart();
                var nownotes = ServerTimeAndNotes[0].Where((data) => time >= Convert.ToSingle(data["StartTime"]) && time < Convert.ToSingle(data["EndTime"])).ToArray();
                var endnotes = ServerTimeAndNotes[0].Where((data) => time >= Convert.ToSingle(data["EndTime"])).ToArray();

                foreach (var endnote in endnotes)
                {
                    Script_NoteHub hub = GameObject.Find("piano").transform.Find("AllNotes").GetComponentsInChildren<Script_NoteHub>().Where((hub) => hub.NoteLevel == Convert.ToInt32(endnote["NoteLevel"])).First();
                    Script_Note note = hub.transform.GetComponentsInChildren<Script_Note>().Where((note) => note.black == Convert.ToBoolean(endnote["Black"]) && note.noteType == (NoteType)Convert.ToInt32(endnote["NoteType"])).First();
                    if (note.NoteDown)
                    {
                        note.NoteDown = false;
                        ServerTimeAndNotes[0].Remove(endnote);
                    }
                }

                foreach (var nownote in nownotes)
                {
                    Script_NoteHub hub = GameObject.Find("piano").transform.Find("AllNotes").GetComponentsInChildren<Script_NoteHub>().Where((hub) => hub.NoteLevel == Convert.ToInt32(nownote["NoteLevel"])).First();
                    Script_Note note = hub.transform.GetComponentsInChildren<Script_Note>().Where((note) => note.black == Convert.ToBoolean(nownote["Black"]) && note.noteType == (NoteType)Convert.ToInt32(nownote["NoteType"])).First();
                    note.NoteDown = true;
                }

                if (ServerTimeAndNotes[0].Count <= 0)
                {
                    TimeStop();
                    ServerTimeAndNotes.RemoveAt(0);
                }
            }
        }
        if (time >= 0) time += Time.deltaTime;
    }
}
