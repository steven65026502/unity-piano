using Newtonsoft.Json;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor.Experimental.RestService;
using UnityEngine;

public class Teleporatation : MonoBehaviour
{
    public void OnClick()
    {
        Manager.Instance.TimeStop();
        string jsonData = JsonConvert.SerializeObject(Manager.Instance.timeAndNotes);
        Debug.Log(jsonData);
        Manager.Instance.timeAndNotes.Clear();
    }
}
