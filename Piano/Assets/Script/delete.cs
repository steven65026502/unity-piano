using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class delete : MonoBehaviour
{
    public TextMeshProUGUI buttonText;

    public void OnClick()
    {
        Manager.Instance.time = 0;
        Manager.Instance.timeAndNotes.Clear();
    }

}
