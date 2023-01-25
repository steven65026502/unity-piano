using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;

public class Script_Note : MonoBehaviour
{

    [SerializeField]
    Data data;
    public class Data
    {
        public float time;
        public NoteType notetype;
    }
    //時間跟音階的陣列
    public List<KeyValuePair<float, NoteType>> timeAndNotes;

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
        timeAndNotes = new List<KeyValuePair<float, NoteType>>();

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
            //儲存時間跟音階
            timeAndNotes.Add(new KeyValuePair<float, NoteType>(elapsedTime, noteType));
            //使用elapsedTime 更新UI 或其他操作    
            elapsedTime += Time.deltaTime;
            timer.text = ((int)elapsedTime).ToString();
            Debug.Log(timeAndNotes);
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


    public void OnClick()
    {

        //抓取timeAndNotes
        List<KeyValuePair<float, NoteType>> timeAndNotes = GetComponent<Script_Note>().timeAndNotes;
        PlayerPrefs.SetString("jsondata", JsonUtility.ToJson(timeAndNotes));
        string jsonData = JsonUtility.ToJson(timeAndNotes);
        File.WriteAllText(Application.dataPath + "/DATA.json", jsonData);
        FileStream fs = new FileStream(Application.dataPath + "/DATA.json", FileMode.Create);
        StreamWriter sw = new StreamWriter(fs);
        sw.WriteLine(data.time);
        sw.WriteLine(data.notetype);
        sw.Close();
        fs.Close();
        if (data == null)
        {
            data = JsonUtility.FromJson<Data>(PlayerPrefs.GetString("jsondata"));
        }
    }
}