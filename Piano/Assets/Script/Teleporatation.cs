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
        string jsonData = JsonConvert.SerializeObject(Manager.timeAndNotes);
        Debug.Log(jsonData);
    }
}
