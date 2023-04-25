using Newtonsoft.Json;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;

public class Teleporatation : MonoBehaviour
{
    public TextMeshProUGUI buttonText;
    public bool returnMode = false;

    public void OnClick()
    {
        bool mode = !returnMode;

        if (!returnMode)
        {
            // 將所有的 hide note 設為 false
            foreach (Script_NoteHub noteHub in FindObjectsOfType<Script_NoteHub>())
            {
                for (int i = 0; i < Script_NoteHub.NoteLength; i++)
                {
                    noteHub.HideNote[i] = false;
                }
            }
            for (int i = 0; i < 9; i++)
            {
                Script_NoteHub noteHub0 = GameObject.Find("NoteHub0").GetComponent<Script_NoteHub>();
                noteHub0.HideNote[i] = true;
            }
            for (int i = 11; i > 1 ; i--)
            {
                Script_NoteHub noteHub8 = GameObject.Find("NoteHub8").GetComponent<Script_NoteHub>();
                noteHub8.HideNote[i] = true;
            }

            Manager.Instance.TimeStop();
            string jsonData = JsonConvert.SerializeObject(Manager.Instance.timeAndNotes);
            Debug.Log(jsonData);
            Manager.Instance.timeAndNotes.Clear();

            // 改變 Main Camera 的 X 軸座標
            Camera.main.transform.position = new Vector3(4.1f, Camera.main.transform.position.y, Camera.main.transform.position.z);

            Camera.main.fieldOfView = 70;

            buttonText.text = "Return";
            returnMode = true;
        }
        else
        {
            for (int i = 0; i <= 3; i++)
            {
                Script_NoteHub noteHub = GameObject.Find($"NoteHub{i}").GetComponent<Script_NoteHub>();
                for (int j = 0; j < Script_NoteHub.NoteLength; j++)
                {
                    noteHub.HideNote[j] = true;
                }
            }

            for (int i = 6; i <= 8; i++)
            {
                Script_NoteHub noteHub = GameObject.Find($"NoteHub{i}").GetComponent<Script_NoteHub>();
                for (int j = 1; j < Script_NoteHub.NoteLength; j++)
                {
                    noteHub.HideNote[j] = true;
                }
            }

            Script_NoteHub noteHub7 = GameObject.Find("NoteHub7").GetComponent<Script_NoteHub>();
            noteHub7.HideNote[0] = true;
            Script_NoteHub noteHub8 = GameObject.Find("NoteHub8").GetComponent<Script_NoteHub>();
            noteHub8.HideNote[0] = true;

            Manager.Instance.TimeStop();
            string jsonData = JsonConvert.SerializeObject(Manager.Instance.timeAndNotes);
            Debug.Log(jsonData);
            Manager.Instance.timeAndNotes.Clear();

            // 改變 Main Camera 的 X 軸座標
            Camera.main.transform.position = new Vector3(-5.3f, Camera.main.transform.position.y, Camera.main.transform.position.z);

            Camera.main.fieldOfView = 24;

            buttonText.text = "Teleportation";
            returnMode = false;
        }
    }
}
