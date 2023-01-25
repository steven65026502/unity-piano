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

    bool mousedown = false;
    public float elapsedTime { get; private set; } = 0;

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
    }
    void Update()
    {
        if(mousedown)
        {   
            //使用elapsedTime 更新UI 或其他操作    
            elapsedTime += Time.deltaTime;
            timer.text = ((int)elapsedTime).ToString();
        }
    }

    private void OnMouseDown()
    {
        Note.Play();
        transform.Translate(0, -0.5f, 0);
        mousedown = true;
        //開始的時間
        elapsedTime = 0;
    }

    private void OnMouseUp()
    {
        //儲存時間跟音階
        Manager.timeAndNotes.Add(new Dictionary<string, object>() { { "time", elapsedTime }, { "NoteType", noteType }, { "Black", black }, { "NoteLevel", transform.parent.GetComponent<Script_NoteHub>().NoteLevel } });
        transform.Translate(0, 0.5f, 0);
        mousedown = false;
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