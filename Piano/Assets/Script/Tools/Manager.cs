using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Linq;
using UnityEngine;
using TMPro;

public class Manager : MonoBehaviour
{
    public const int INT_LEN = 4;
    public static bool returnMode = false;
    private static Manager _instance = null;

    object responselock = new object();

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
    public PianoRoll timeAndNotes;
    public List<PianoRoll> ServerTimeAndNotes;
    private Socket socket;
    public float time { get; set; } = -1;

    private void Awake()
    {
        time = -1;
        doinupdate = new List<Action>();
        timeAndNotes = new PianoRoll();
        ServerTimeAndNotes = new List<PianoRoll>();
        socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        socket.Bind(new IPEndPoint(IPAddress.Loopback, 1234));
        socket.Listen(-1);
        socket.BeginAccept(AcceptCallback, null);
    }

    private void AcceptCallback(IAsyncResult asyncCallback)
    {
        Socket client = socket.EndAccept(asyncCallback);
        Debug.Log(client.RemoteEndPoint.ToString() + " Connected");
        byte[] len = new byte[4];
        client.BeginReceive(len, 0, INT_LEN, SocketFlags.None, Response, new object[] { client, len });
        socket.BeginAccept(AcceptCallback, null);
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
        Socket client = (Socket)((object[])asyncResult.AsyncState)[0];
        byte[] lendata = (byte[])((object[])asyncResult.AsyncState)[1];
        client.EndReceive(asyncResult);
        if (lendata.Length < 1)
        {
            client.Close();
            return;
        }
        lock (responselock)
        {
            int len = System.BitConverter.ToInt32(HToNAndNToH(lendata));
            byte[] data = new byte[len];
            if (client.Receive(data, 0, len, SocketFlags.None) < 1)
            {
                client.Close();
                return;
            }
            lock (ServerTimeAndNotes)
            {
                if (ServerTimeAndNotes.Count <= 0)
                    TimeStop();
                ServerTimeAndNotes.Add(JsonConvert.DeserializeObject<PianoRoll>(System.Text.Encoding.UTF8.GetString(data)));
            }
            client.BeginReceive(lendata, 0, INT_LEN, SocketFlags.None, Response, lendata);
        }
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
                if (ServerTimeAndNotes.Count > 0)
                {
                    if (!returnMode)
                    {
                        Teleporatation teleportation = GameObject.FindObjectOfType<Teleporatation>();
                        teleportation.OnClick();
                        returnMode = true;
                    }
                }
                else
                {
                    returnMode = false;
                }
                TimeStart();
                var nownotes = ServerTimeAndNotes[0].onset_events.Where((data) => time >= Convert.ToSingle(data[(byte)PianoRollOnsetEventIndex.StartTime] / PianoRoll.sectobit) && time < Convert.ToSingle(data[(byte)PianoRollOnsetEventIndex.EndTime] / PianoRoll.sectobit)).ToArray();
                var endnotes = ServerTimeAndNotes[0].onset_events.Where((data) => time >= Convert.ToSingle(data[(byte)PianoRollOnsetEventIndex.EndTime] / PianoRoll.sectobit)).ToArray();
                foreach (var endnote in endnotes)
                {
                    ServerTimeAndNotes[0].onset_events.Remove(endnote);
                }

                foreach (var nownote in nownotes)
                {
                    Script_Note note = PianoRoll.OnsetEventToNote(GameObject.Find("piano").transform.Find("AllNotes").GetComponentsInChildren<Script_NoteHub>(), nownote);
                    //if (note.NoteDown) Debug.Log(note.NoteString + " down");
                    note.Note.volume = PianoRoll.PowerToVolume(nownote[(byte)PianoRollOnsetEventIndex.Power]);
                    note.SetPianoEvent(nownote);
                    ServerTimeAndNotes[0].onset_events.Remove(nownote);
                }

                if (ServerTimeAndNotes[0].onset_events.Count <= 0)
                {
                    TimeStop();
                    ServerTimeAndNotes.RemoveAt(0);
                }
                else
                {
                    TextMeshProUGUI timer = GameObject.Find("timer").transform.Find("timer_text").GetComponent<TextMeshProUGUI>();
                    timer.text = ((int)Instance.time).ToString();
                }
            }
        }
        if (time >= 0) time += Time.deltaTime;
    }
}
