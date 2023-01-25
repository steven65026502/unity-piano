using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraRay : MonoBehaviour
{   

    void Update()
    {
        
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);

        RaycastHit hit;

        if (Input.GetMouseButton(0) && Physics.Raycast(ray,out hit))
        {
            Debug.DrawLine(Camera.main.transform.position,hit.transform.position,Color.red,10f,true);
        }

    }
}

