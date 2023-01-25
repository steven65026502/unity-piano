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
        Manager.Instance.timeAndNotes.Clear();
        string jsonData = JsonConvert.SerializeObject(Manager.Instance.timeAndNotes);
        Debug.Log(jsonData);
    }
}
