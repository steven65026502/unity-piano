using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using System;

public class Script_Note : MonoBehaviour
{
    //時間跟音階的陣列
    public AudioSource Note { get; private set; }
    private TextMeshProUGUI text;
    private TextMeshProUGUI timer;

    private int[] pianoevent = null;

    public bool black = false;
    public NoteType noteType;

    bool mouseon = false;

    private bool _notedown = false;
    bool NoteDown 
    { 
        get
        {
            return _notedown;
        }
        set
        {
            if(value)
            {
                if (!NoteDown)
                {
                    _notedown = true;
                    Note.time = 0;
                    Note.Play();
                    transform.Translate(0, -0.5f, 0);
                    //開始的時間
                    if (Manager.Instance.time < 0) Manager.Instance.TimeStart();
                    StartTime = Manager.Instance.time;
                }
            }
            else
            {
                if (NoteDown)
                {
                    _notedown = false;
                    EndTime = Manager.Instance.time;
                    //儲存時間跟音階
                    Manager.Instance.timeAndNotes.onset_events.Add(PianoRoll.NoteToOnsetEvent(this));
                    transform.Translate(0, 0.5f, 0);
                }
            }
        }
    }
    
    public string NoteString
    {
        get
        {
            return string.Format("{0}{1}{2}", black ? "#" : "", ((NoteName)noteType).ToString(), transform.parent.GetComponent<Script_NoteHub>().NoteLevel);
        }
    }

    public float StartTime { get; private set; } = 0;
    public float EndTime { get; private set; } = 0;

    public void Awake()
    {
        //陣列初始化
        //timeAndNotes = new List<Dictionary<string, object>>();

        timer = GameObject.Find("timer").transform.Find("timer_text").GetComponent<TextMeshProUGUI>();
        text = transform.parent.parent.parent.Find("Canvas").Find("nowtype").GetComponent<TextMeshProUGUI>();

        Note = GetComponent<AudioSource>();                 //抓取音檔
        Note.clip = Resources.Load<AudioClip>(string.Format("NoteSound/{0}{1}{2}", noteType.ToString(), black ? "b" : "", transform.parent.GetComponent<Script_NoteHub>().NoteLevel));
    }

    public void Start()
    {
        Manager.Instance.timeAndNotes.Clear();
    }
    void Update()
    {
        if(NoteDown)
        {   
            if(mouseon) timer.text = ((int)(Manager.Instance.time - StartTime)).ToString();
            if (Note.isPlaying && Note.time > 1.7f) Note.time = 1.5f; //維持在音檔為1.6秒的狀態
        }
        if (pianoevent != null && (Manager.Instance.time >= Convert.ToSingle(pianoevent[(byte)PianoRollOnsetEventIndex.EndTime] / PianoRoll.sectobit) || Manager.Instance.time < 0))
        {
            NoteDown = false;
            pianoevent = null;
        }
    }

    public void SetPianoEvent(int[] nowevent)
    {
        NoteDown = false;
        pianoevent = nowevent;
        NoteDown = true;
    }

    private void OnMouseDown()
    {
        mouseon = true;
        NoteDown = true;
    }

    private void OnMouseUp()
    {
        mouseon = false;
        NoteDown = false;
    }

    public void OnMouseOver() 
    {
        text.text = NoteString;
    }

    public void OnMouseExit() 
    {
        if(text.text == NoteString)
            text.text = "";
    }
}