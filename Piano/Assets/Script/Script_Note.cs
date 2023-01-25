using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using Newtonsoft.Json;

public class Script_Note : MonoBehaviour
{
    //時間跟音階的陣列
    private AudioSource Note;
    private TextMeshProUGUI text;
    private TextMeshProUGUI timer;

    public bool black = false;
    public NoteType noteType;

    private bool _notedown = false;
    public bool NoteDown 
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
                    Manager.Instance.timeAndNotes.Add(new Dictionary<string, object>() { { "StartTime", StartTime }, { "EndTime", EndTime }, { "NoteType", noteType }, { "Black", black }, { "NoteLevel", transform.parent.GetComponent<Script_NoteHub>().NoteLevel } });
                    transform.Translate(0, 0.5f, 0);
                }
            }
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

        Note = GetComponent<AudioSource>();
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
            //使用elapsedTime 更新UI 或其他操作    
            timer.text = ((int)(Manager.Instance.time - StartTime)).ToString();
            if (Note.isPlaying && Note.time > 1.6f) Note.time = 1.6f;
        }
    }

    private void OnMouseDown()
    {
        NoteDown = true;
    }

    private void OnMouseUp()
    {
        NoteDown = false;
    }

    public void OnMouseOver() 
    {
        text.text = string.Format("{0}{1}{2}", black ? "#" : "", ((NoteName)noteType).ToString(), transform.parent.GetComponent<Script_NoteHub>().NoteLevel);
    }

    public void OnMouseExit() 
    {
        if(text.text == string.Format("{0}{1}{2}", black ? "#" : "", ((NoteName)noteType).ToString(), transform.parent.GetComponent<Script_NoteHub>().NoteLevel))
            text.text = "";
    }
}