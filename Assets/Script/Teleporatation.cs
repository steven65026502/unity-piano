using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor.Experimental.RestService;
using UnityEngine;

public class Teleporatation : MonoBehaviour
{
    [SerializeField]
    Data data;
    public class Data
    {
        public float time;
        public NoteType notetype;
    }
    public void OnClick()
    {
        //抓取timeAndNotes
        List<KeyValuePair<float, NoteType>> timeAndNotes = GetComponent<Script_Note>().timeAndNotes;

        //將陣列轉換成JSON字符串
        string jsonData = JsonUtility.ToJson(timeAndNotes);
        //儲存JSON字符串到文件中
        File.WriteAllText(Application.persistentDataPath + "/data.json", jsonData);
    }

    /*public void OnClick()
    {
        List<KeyValuePair<float, NoteType>> timeAndNotes = GetComponent<Script_Note>().timeAndNotes;
        PlayerPrefs.SetString("jsondata", JsonUtility.ToJson(timeAndNotes)); 
        
        FileStream fs = new FileStream(Application.dataPath + "/DATA.json", FileMode.Create);
        StreamWriter sw = new StreamWriter(fs);
        sw.WriteLine(data.time);
        sw.WriteLine(data.notetype);
        sw.Close();
        fs.Close();
        if(data == null)
        {
            data = JsonUtility.FromJson<Data>(PlayerPrefs.GetString("jsondata"));
        }
    }*/


}
