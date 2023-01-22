using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
public class Script_Note : MonoBehaviour
{
    private AudioSource Note;
    private TextMeshProUGUI text;
    private TextMeshProUGUI timer;

    public bool black = false;
    public NoteType noteType;

    //計時器參數
    bool mousedown = false;
    public float elapsedTime { get; private set; } = 0;

    public void Awake()
    {
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
        Note.Stop();
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