using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

[ExecuteInEditMode]
public class Script_NoteHub : MonoBehaviour
{
    public const int NoteLength = 12;
    public int NoteLevel = 0;
    public Script_Note[] Notes = new Script_Note[NoteLength];
    public bool[] HideNote = new bool[NoteLength];

    private void Update()
    {
        for(int i = 0; i < NoteLength; i++) 
        {
            foreach(MeshRenderer meshRenderer in Notes[i].GetComponents<MeshRenderer>())
                meshRenderer.enabled = !HideNote[i];
            foreach (BoxCollider boxCollider in Notes[i].GetComponents<BoxCollider>())
                boxCollider.enabled = !HideNote[i];
            foreach (MeshRenderer meshRenderer in Notes[i].GetComponentsInChildren<MeshRenderer>())
                meshRenderer.enabled = !HideNote[i];
            foreach (BoxCollider boxCollider in Notes[i].GetComponentsInChildren<BoxCollider>())
                boxCollider.enabled = !HideNote[i];
        }
    }
}
